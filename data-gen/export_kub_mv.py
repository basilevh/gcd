'''
Created by Basile Van Hoorick for GCD, 2024.
Generates multi-view Kubric videos.
'''

import os  # noqa
import sys  # noqa

sys.path.insert(0, os.path.join(os.getcwd(), 'kubric/'))  # noqa
sys.path.insert(0, os.path.join(os.getcwd()))  # noqa

# Library imports.
import argparse
import collections
import collections.abc
import colorsys
import copy
import datetime
import glob
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import pickle
import platform
import random
import shutil
import sys
import time
import warnings
from collections import defaultdict

import cv2
import fire
import imageio
import joblib
import lovely_numpy
import lovely_tensors
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import rich.console
import rich.logging
import rich.progress
import scipy
import tqdm
import tqdm.rich
from einops import rearrange, repeat
from lovely_numpy import lo
from rich import print
from tqdm import TqdmExperimentalWarning

# Internal imports.
import data_utils

np.set_printoptions(precision=3, suppress=True)


def main(root_dp='/path/to/kubric_mv_dbg',
         mass_est_fp='gpt_mass_v4.txt',
         num_scenes=100, start_idx=0, end_idx=99999,
         num_workers=16, restart_count=30, seed=400,
         num_perturbs=1, num_views=16, frame_width=384, frame_height=256,
         num_frames=30, frame_rate=12, motion_blur=0, save_depth=1, save_coords=1, save_bkfw=0,
         render_samples_per_pixel=16, render_use_gpu=0, max_camera_speed=8.0,
         focal_length=32.0, fixed_alter_poses=1, few_views=4):

    # ==================================
    # CUSTOMIZE DATASET PARAMETERS HERE:

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    perturbs_first_scenes = 0  # Only test.
    views_first_scenes = 999999  # Only test.
    test_first_scenes = 0  # For handling background & object asset splits (optional).

    root_dn = os.path.basename(root_dp)
    ignore_if_exist = True  # Already generated scene folders will be skipped.

    min_static = 6  # Kubric / MOVi = 10.
    max_static = 16  # Kubric / MOVi = 20.
    min_dynamic = 1  # Kubric / MOVi = 1.
    max_dynamic = 6  # Kubric / MOVi = 3.

    camera_radius_range = [12.0, 16.0]  # In meters.
    static_diameter_range = [1.0, 2.75]  # Kubric / MOVi = [0.75, 3.0].
    dynamic_diameter_range = [1.0, 2.75]  # Kubric / MOVi = [0.75, 3.0].

    fixed_radius = 15.0  # In meters.
    fixed_elevation_many = 5  # Primary target distribution.
    fixed_elevation_few = 45  # Look inside containers / behind occluders.
    fixed_look_at = [0.0, 0.0, 1.0]  # Matches _setup_camera().

    # NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
    # directory inbetween runs. This counter indicates when all threads should finish.
    total_scn_cnt = mp.Value('i', 0)

    os.makedirs(root_dp, exist_ok=True)

    def do_scene(worker_idx, scene_idx, scene_dp, scene_dn):

        # Assign resources (which CPU and/or GPU I can use).
        data_utils.update_os_cpu_affinity(scene_idx % 4, 4)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_idx % 8)

        # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
        import kubric as kb
        import kubric_sim
        import pybullet as pb

        render_cpu_threads = int(np.ceil(mp.cpu_count() / max(num_workers, 4)))
        print(f'{worker_idx}: {scene_idx}: Using {render_cpu_threads} CPU threads for rendering.')

        np.random.seed(seed + scene_idx * 257)
        scratch_dir = f'/tmp/mygenkub_{root_dn}/{scene_idx:05d}_{np.random.randint(10000, 99999)}'

        # NOTE: This instance must only be created once per process!
        my_kubric = kubric_sim.MyKubricSimulatorRenderer(
            frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
            frame_rate=frame_rate, motion_blur=motion_blur,
            render_samples_per_pixel=render_samples_per_pixel, render_use_gpu=render_use_gpu,
            render_cpu_threads=render_cpu_threads, scratch_dir=scratch_dir, mass_est_fp=mass_est_fp,
            max_camera_speed=max_camera_speed, mass_scaling_law=2.0)

        os.makedirs(scene_dp, exist_ok=True)

        start_time = time.time()

        phase = 'test' if scene_idx < test_first_scenes else 'train'
        t = my_kubric.prepare_next_scene(
            phase, seed + scene_idx, camera_radius_range=camera_radius_range,
            focal_length=focal_length)
        print(f'{worker_idx}: {scene_idx}: prepare_next_scene took {t:.2f}s')

        t = my_kubric.insert_static_objects(
            min_count=min_static, max_count=max_static, any_diameter_range=static_diameter_range)
        print(f'{worker_idx}: {scene_idx}: insert_static_objects took {t:.2f}s')

        beforehand = int(round(4 * frame_rate))
        (_, _, t) = my_kubric.simulate_frames(-beforehand, -1)
        print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')

        t = my_kubric.reset_objects_velocity_friction_restitution()
        print(f'{worker_idx}: {scene_idx}: '
              f'reset_objects_velocity_friction_restitution took {t:.2f}s')

        t = my_kubric.insert_dynamic_objects(
            min_count=min_dynamic, max_count=max_dynamic, any_diameter_range=dynamic_diameter_range)
        print(f'{worker_idx}: {scene_idx}: insert_dynamic_objects took {t:.2f}s')

        all_data_stacks = []
        all_videos = []

        # Determine multiplicity of this scene based on index.
        used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
        used_num_views = num_views if scene_idx < views_first_scenes else 1
        start_yaw = my_kubric.random_state.uniform(0.0, 360.0)

        # Loop over butterfly effect variations.
        for perturb_idx in range(used_num_perturbs):

            print()
            print(f'{worker_idx}: {scene_idx}: '
                  f'perturb_idx: {perturb_idx} / used_num_perturbs: {used_num_perturbs}')
            print()

            # Ensure that the simulator resets its state for every perturbation.
            if perturb_idx == 0 and used_num_perturbs >= 2:
                print(f'Saving PyBullet simulator state...')
                # https://github.com/bulletphysics/bullet3/issues/2982
                pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
                pb_state = pb.saveState()

            elif perturb_idx >= 1:
                print(f'{worker_idx}: {scene_idx}: Restoring PyBullet simulator state...')
                pb.restoreState(pb_state)

            # Always simulate a little bit just before the actual starting point to ensure Kubric
            # updates its internal state (in particular, object positions) properly.
            (_, _, t) = my_kubric.simulate_frames(-1, 0)
            print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')

            if used_num_perturbs >= 2:
                t = my_kubric.perturb_object_positions(max_offset_meters=0.005)
                print(f'{worker_idx}: {scene_idx}: perturb_object_positions took {t:.2f}s')

            (_, _, t) = my_kubric.simulate_frames(0, num_frames)
            print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')

            # Loop over camera viewpoints.
            for view_idx in range(used_num_views):

                print()
                print(f'{worker_idx}: {scene_idx}: '
                      f'view_idx: {view_idx} / used_num_views: {used_num_views}')
                print()

                if fixed_alter_poses:
                    if view_idx < few_views:
                        # Sparse high up.
                        azimuth_deg = view_idx * 360.0 / few_views
                        elevation_deg = fixed_elevation_few

                    else:
                        # Dense low down.
                        azimuth_deg = (view_idx - few_views + 0.5) * 360.0 / (num_views - few_views)
                        elevation_deg = fixed_elevation_many

                    azimuth_rad = np.deg2rad(azimuth_deg)
                    elevation_rad = np.deg2rad(elevation_deg)

                    camera_position = np.array([np.cos(azimuth_rad) * np.cos(elevation_rad),
                                                np.sin(azimuth_rad) * np.cos(elevation_rad),
                                                np.sin(elevation_rad)]) * fixed_radius
                    camera_position += fixed_look_at  # To preserve desired viewing angles.

                    print(f'{worker_idx}: {scene_idx}: '
                          f'Applying exact camera position: {camera_position} '
                          f'and look_at: {fixed_look_at}...')
                    t = my_kubric.setup_camera_exact(
                        camera_position, fixed_look_at, focal_length=focal_length)

                else:
                    camera_yaw = view_idx * 360.0 / used_num_views + start_yaw
                    print(f'{worker_idx}: {scene_idx}: '
                          f'Applying initial camera yaw: {camera_yaw:.1f} degrees...')
                    t = my_kubric.set_camera_yaw(camera_yaw)
                    print(f'{worker_idx}: {scene_idx}: set_camera_yaw took {t:.2f}s')

                # NOTE: Let's actually keep object_coordinates for now.
                # (data_stack, t) = my_kubric.render_frames(
                #     0, num_frames - 1, return_layers=['rgba', 'forward_flow', 'depth', 'normal',
                #                                       'object_coordinates', 'segmentation'])
                # (data_stack, t) = my_kubric.render_frames(
                #     0, num_frames - 1, return_layers=['rgba', 'depth', 'segmentation'])

                save_layers = ['rgba', 'forward_flow', 'normal', 'segmentation']
                if save_depth:
                    save_layers.append('depth')
                if save_coords:
                    save_layers.append('object_coordinates')
                if save_bkfw:
                    save_layers.append('backward_flow')
                (data_stack, t) = my_kubric.render_frames(
                    0, num_frames - 1, return_layers=save_layers)
                print(f'{worker_idx}: {scene_idx}: render_frames took {t:.2f}s')

                rgb = data_stack['rgba'][..., 0:3]  # (T, H, W, 3) of uint8 in [0, 255].
                if save_depth:
                    depth = data_stack['depth'][..., 0]  # (T, H, W) of float32 in (0, inf).
                    # (T, H, W, 3) of uint8 in [0, 255].
                    depth_vis = data_utils.depth_to_rgb_vis(depth)
                segm = data_stack['segmentation'][..., 0]  # (T, H, W) of uint8 in [0, K].
                segm_vis = data_utils.segm_ids_to_rgb(segm, hue_step=0.04)
                # (T, H, W, 3) of uint8 in [0, 255].
                (T, H, W, C) = rgb.shape

                (metadata, t) = my_kubric.get_metadata(exclude_collisions=view_idx > 0)
                print(f'{worker_idx}: {scene_idx}: get_metadata took {t:.2f}s')

                # Create videos of source data and annotations.
                fn_prefix = f'{scene_dn}_p{perturb_idx}_v{view_idx}'
                data_utils.save_video(
                    os.path.join(scene_dp, f'{fn_prefix}_rgb.mp4'), rgb, frame_rate, 9)
                if save_depth:
                    data_utils.save_video(
                        os.path.join(scene_dp, f'{fn_prefix}_depth.mp4'), depth_vis, frame_rate, 9)
                data_utils.save_video(
                    os.path.join(scene_dp, f'{fn_prefix}_segm.mp4'), segm_vis, frame_rate, 9)

                # Write all individual frames for normal and divided videos.
                dst_dp = os.path.join(scene_dp, f'frames_p{perturb_idx}_v{view_idx}')
                print(f'{worker_idx}: {scene_idx}: '
                      f'Saving all generated frames to: {dst_dp}...')
                t = my_kubric.write_all_data(dst_dp)
                print(f'{worker_idx}: {scene_idx}: write_all_data took {t:.2f}s')

                # Write metadata last (this is a marker of completion).
                dst_json_fp = os.path.join(scene_dp, f'{fn_prefix}.json')
                kb.write_json(metadata, dst_json_fp)

                all_videos.append(rgb)

                print(f'{worker_idx}: {scene_idx}: '
                      f'All together took {time.time() - start_time:.2f}s')

            pass

        # Visualize pixel deltas across pairs of perturbations.
        if used_num_perturbs >= 2:
            for perturb_idx in range(used_num_perturbs):
                for view_idx in range(used_num_views):
                    p1_idx = perturb_idx
                    p2_idx = (perturb_idx + 1) % used_num_perturbs
                    rgba_p1 = all_data_stacks[view_idx + p1_idx * used_num_views]['rgba']
                    rgba_p2 = all_data_stacks[view_idx + p2_idx * used_num_views]['rgba']
                    rgb_delta = rgba_p1[..., :3].astype(
                        np.int16) - rgba_p2[..., :3].astype(np.int16)
                    rgb_delta = np.clip(np.abs(rgb_delta * 2), 0, 255).astype(np.uint8)

                    data_utils.save_video(os.path.join(
                        scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_delta.mp4'),
                        rgb_delta, frame_rate, 9)

                    all_videos.append(rgb_delta)

        # Finally, bundle all variations and viewpoints together into one big video.
        big_video = None
        all_videos = np.stack(all_videos, axis=0)
        if len(all_videos) == used_num_perturbs * used_num_views:
            big_video = rearrange(all_videos, '(P V) T H W C -> T (V H) (P W) C',
                                  P=used_num_perturbs, V=used_num_views)
        elif len(all_videos) == 2 * used_num_perturbs * used_num_views:
            big_video = rearrange(all_videos, '(D P V) T H W C -> T (V H) (P D W) C',
                                  D=2, P=used_num_perturbs, V=used_num_views)
            # Ignore last ("wrap around") delta video for big bundle.
            big_video = big_video[:, :, :-W]
        else:
            print()
            print(f'{worker_idx}: {scene_idx}: Expected {used_num_perturbs * used_num_views} or '
                  f'{used_num_perturbs * used_num_views * 2} '
                  f'videos, but got {len(all_videos)}?')
            print()

        if big_video is not None:
            data_utils.save_video(
                os.path.join(scene_dp, f'{scene_dn}_bundle.mp4'), big_video, frame_rate, 7)

        print()

        pass

    def worker(worker_idx, num_workers, total_scn_cnt):

        machine_name = platform.node()
        log_name = f'{root_dn}_{machine_name}'

        my_start_idx = worker_idx + start_idx
        my_end_idx = min(num_scenes, end_idx)

        for scene_idx in range(my_start_idx, my_end_idx, num_workers):

            scene_dn = f'scn{scene_idx:05d}'
            scene_dp = os.path.join(root_dp, scene_dn)

            print()
            print(f'{worker_idx}: scene_idx: {scene_idx} / scene_dn: {scene_dn}')
            print()

            # Determine multiplicity of this scene based on index.
            used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
            used_num_views = num_views if scene_idx < views_first_scenes else 1

            # Check for the latest file that could have been written.
            dst_json_fp = os.path.join(
                scene_dp, f'{scene_dn}_p{used_num_perturbs - 1}_v{used_num_views - 1}.json')
            if ignore_if_exist and os.path.exists(dst_json_fp):
                print(f'{worker_idx}: This scene already exists at {dst_json_fp}, skipping!')
                continue

            else:
                total_scn_cnt.value += 1
                print(f'{worker_idx}: Total scene counter: '
                      f'{total_scn_cnt.value} / {restart_count}')
                if total_scn_cnt.value >= restart_count:
                    print()
                    print(f'{worker_idx}: Reached max allowed scene count, exiting!')
                    print()
                    break

                # We perform the actual generation in a separate thread to try to ensure that
                # no memory leaks survive.
                p = mp.Process(target=do_scene, args=(
                    worker_idx, scene_idx, scene_dp, scene_dn))
                p.start()
                p.join()

            pass

        print()
        print(f'I am done!')
        print()

        pass

    if num_workers <= 1:

        worker(0, 1, total_scn_cnt)

    else:

        processes = [mp.Process(target=worker,
                                args=(worker_idx, num_workers, total_scn_cnt))
                     for worker_idx in range(num_workers)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    pass


if __name__ == '__main__':

    fire.Fire(main)

    pass
