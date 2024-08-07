'''
Created by Basile Van Hoorick for GCD, 2024.
Computes & caches merged point clouds for efficient ParallelDomain-4D data loading.
'''

import os  # noqa
import sys  # noqa

# Library imports.
import argparse
import cv2
import fire
import glob
import imageio
import joblib
import json
import lovely_numpy
import lovely_tensors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pathlib
import PIL
import platform
import rich
import rich.progress
import time
import torch
import torchvision
import torchvision.transforms.functional
import tqdm
import tqdm.rich
import warnings
from einops import rearrange
from lovely_numpy import lo
from pyquaternion import Quaternion
from rich import print
from tqdm import TqdmExperimentalWarning

# Internal imports.
import data_utils

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

process_dtype = torch.float32
store_xyz_dtype = torch.float16

MAX_DEPTH = 30000.0  # 30 kilometers; just below half of torch float16 max value of 65504.


# NOTE: 40 out of 50 means at least 80% of PD data is available.
def is_finished(output_dp, expect_files):
    exist_files = len(glob.glob(os.path.join(output_dp, '*.pt')))
    return exist_files >= expect_files


def get_intrinsics_matrix(intrinsics_dict):
    intrinsics_matrix = torch.tensor(
        [[intrinsics_dict['fx'], 0.0, intrinsics_dict['cx']],
         [0.0, intrinsics_dict['fy'], intrinsics_dict['cy']],
         [0.0, 0.0, 1.0]], dtype=process_dtype)
    return intrinsics_matrix


def get_extrinsics_matrix(extrinsics_dict):
    rot_q = extrinsics_dict['rotation']
    rot_t = extrinsics_dict['translation']
    rot_m = Quaternion(rot_q['qw'], rot_q['qx'], rot_q['qy'], rot_q['qz']).rotation_matrix
    extrinsics_matrix = torch.eye(4, dtype=process_dtype)
    extrinsics_matrix[0:3, 0:3] = torch.tensor(rot_m)
    extrinsics_matrix[0:3, 3] = torch.tensor([rot_t['x'], rot_t['y'], rot_t['z']])
    return extrinsics_matrix


def process_example(worker_idx, gpu_idx, example,
                    sel_views, sel_frames, sel_modals, debug, ignore_if_exist):
    import torch

    (scene_dp, output_dp) = example

    if ignore_if_exist and is_finished(output_dp, 40):
        print(f'[yellow]{worker_idx}: Skipping {output_dp} because it already contains '
              f'40 or more files...')
        return False

    device = torch.device(f'cuda:{gpu_idx}')

    # NOTE: To save memory, we do not retain large arrays or tensors in memory over time!
    print()
    print(f'[yellow]{worker_idx}: Processing scene {example}...')
    print(f'[yellow]{worker_idx}: Unprojecting all pixels to points...')
    print(f'[yellow]{worker_idx}: Saving merged point clouds to disk...')
    start_time = time.time()

    scene_dn = os.path.basename(scene_dp)
    rgb_dp = os.path.join(scene_dp, 'rgb')
    depth_dp = os.path.join(scene_dp, 'depth')
    segm_dp = os.path.join(scene_dp, 'semantic_segmentation_2d')
    # inst_dp = os.path.join(scene_dp, 'instance_segmentation_2d')

    # Typically camera0 - camera15, yaw-0, yaw-60, and yaw-neg-60.
    avail_views = np.array(sorted(os.listdir(rgb_dp)))
    used_views = avail_views[sel_views]
    print(f'[gray]{worker_idx}: avail_views: {avail_views}')
    if len(used_views) < len(avail_views):
        print(f'[gray]{worker_idx}: sel_views: {sel_views}')
        print(f'[gray]{worker_idx}: used_views: {used_views}')

    calibration_fp = glob.glob(os.path.join(scene_dp, 'calibration', '*.json'))[0]
    calibration = data_utils.load_json(calibration_fp)
    ontology_fp = glob.glob(os.path.join(scene_dp, 'ontology', '*.json'))[0]
    ontology = data_utils.load_json(ontology_fp)

    # NOTE: Camera parameters do not vary over time in this dataset.
    all_intrinsics = dict()  # Maps view_name to (3, 3) tensor of float.
    all_extrinsics = dict()  # Maps view_name to (4, 4) tensor of float.

    for (view_name, intrinsics_dict, extrinsics_dict) in zip(
            calibration['names'], calibration['intrinsics'], calibration['extrinsics']):
        if not (view_name in used_views):
            continue

        # NOTE: Unlike Kubric, the intrinsics matrix is already unnormalized (pixel coordinates).
        intrinsics_matrix = get_intrinsics_matrix(intrinsics_dict)  # (3, 3) tensor of float.
        extrinsics_matrix = get_extrinsics_matrix(extrinsics_dict)  # (4, 4) tensor of float.
        all_intrinsics[view_name] = intrinsics_matrix.to(device)
        all_extrinsics[view_name] = extrinsics_matrix.to(device)

    all_intrinsics = torch.stack([all_intrinsics[view_name] for view_name in used_views], dim=0)
    # (V, 3, 3) tensor of float.
    all_extrinsics = torch.stack([all_extrinsics[view_name] for view_name in used_views], dim=0)
    # (V, 4, 4) tensor of float.

    for (i, t) in tqdm.tqdm(enumerate(sel_frames)):

        # For this frame, load data from all viewpoints.
        frame_rgb = []
        frame_depth = []
        frame_segm = []
        skip_frame = False

        for (j, view_name) in enumerate(used_views):

            cur_rgb_dp = os.path.join(rgb_dp, view_name)
            cur_depth_dp = os.path.join(depth_dp, view_name)
            cur_segm_dp = os.path.join(segm_dp, view_name)
            rgb_fp = os.path.join(cur_rgb_dp, f'{t * 10 + 5:018d}.png')
            depth_fp = os.path.join(cur_depth_dp, f'{t * 10 + 5:018d}.npz')
            segm_fp = os.path.join(cur_segm_dp, f'{t * 10 + 5:018d}.png')

            if not (os.path.isfile(rgb_fp) and os.path.isfile(depth_fp) and os.path.isfile(segm_fp)):
                print(f'[orange3]{worker_idx}: Missing {rgb_fp} or {depth_fp} or {segm_fp}, '
                      f'skipping this frame only...')
                skip_frame = True
                break

            rgb = plt.imread(rgb_fp)[..., 0:3]
            rgb = torch.tensor(rgb, dtype=torch.float32, device=device)
            # (H, W, 3) tensor of float32 in [0, 1].

            depth = np.load(depth_fp)['data']
            depth = torch.tensor(depth, dtype=torch.float32, device=device)
            depth = torch.clamp(depth, 0.0, MAX_DEPTH)
            # (H, W) tensor of float32 in [0, inf).

            # NOTE: Max id is 255 = Void.
            segm = plt.imread(segm_fp)
            segm = torch.tensor(segm[..., 0] * 255.0, dtype=torch.uint8, device=device)
            # (H, W) tensor of uint8 in [0, 255].

            frame_rgb.append(rgb)
            frame_depth.append(depth)
            frame_segm.append(segm)

        if skip_frame:
            continue

        frame_rgb = torch.stack(frame_rgb, dim=0)  # (V, H, W, 3) tensor of float32.
        frame_depth = torch.stack(frame_depth, dim=0)  # (V, H, W) tensor of float32.
        frame_segm = torch.stack(frame_segm, dim=0)  # (V, H, W) tensor of uint8.

        # For this frame, deproject & merge point clouds.
        frame_xyzfeats = []

        for (j, v) in enumerate(sel_views):
            assert 'rgb' in sel_modals
            assert 'segm' in sel_modals

            rgb = frame_rgb[j]  # (H, W, 3) tensor of float32.
            segm = frame_segm[j]  # (H, W) tensor of uint8.
            tag = torch.ones_like(segm) * j  # (H, W) tensor of uint8.
            feats = torch.cat([rgb, segm[..., None], tag[..., None]], dim=-1)
            # (H, W, 5) tensor of float32.

            depth = frame_depth[j]  # (H, W) tensor of float.
            K = all_intrinsics[j]  # (3, 3) tensor of float.
            RT = all_extrinsics[j]  # (4, 4) tensor of float.

            xyzfeats = data_utils.unproject_pixels_to_points(feats, depth, K, RT, process_dtype)
            frame_xyzfeats.append(xyzfeats)
            # (N, 3 + C) tensor of float.

        frame_xyzfeats = torch.stack(frame_xyzfeats, dim=0)
        # (V, N, 3 + C) tensor of float.

        # Now compress stuff to save disk space.
        # See https://pytorch.org/docs/stable/tensors.html#id4 -- 10 significand bits.
        # NOTE: I verified these values and it creates a relative error of 0.1% on average.
        # The error is highest (3%) for very small numbers (abs < 1e-5).
        # NOTE: float16 has a max range of +- 65504, which will often overflow to infinity in this
        # dataset if I didn't apply clamping -- this is important for the skybox!
        store_xyz = frame_xyzfeats[..., 0:3].clone().type(store_xyz_dtype)
        store_rgb = (frame_xyzfeats[..., 3:6].clone() * 255.0).type(torch.uint8)
        store_segm = frame_xyzfeats[..., 6:7].clone().type(torch.uint8)
        store_tag = frame_xyzfeats[..., 7:8].clone().type(torch.uint8)
        store_xyz = store_xyz.detach().cpu()
        store_rgb = store_rgb.detach().cpu()
        store_segm = store_segm.detach().cpu()
        store_tag = store_tag.detach().cpu()

        # For this frame, save results to disk.
        os.makedirs(output_dp, exist_ok=True)
        modals_fn = '_'.join(sel_modals)
        dst_fp = os.path.join(output_dp, f'pcl_{modals_fn}_{t * 10 + 5:06d}.pt')
        torch.save([store_xyz, store_rgb, store_segm, store_tag], dst_fp)

    print(f'[magenta]{worker_idx}: Unprojection & saving took {time.time() - start_time:.2f}s')

    return True


def worker_fn(worker_idx, gpu_idx, num_workers, my_examples,
              sel_views, sel_frames, sel_modals, debug, ignore_if_exist):

    # Only now can we import torch.
    import torch
    torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)

    # Update CPU affinity.
    data_utils.update_os_cpu_affinity(worker_idx, num_workers)

    # Start iterating over all videos.
    if debug:
        to_loop = tqdm.tqdm(list(enumerate(my_examples)))
    else:
        to_loop = tqdm.rich.tqdm(list(enumerate(my_examples)))

    for (i, example) in to_loop:
        try:
            process_example(worker_idx, gpu_idx, example,
                            sel_views, sel_frames, sel_modals, debug, ignore_if_exist)
        except Exception as e:
            print(f'[red]{worker_idx}: ERROR in {example}: {e}')
            continue

    print()
    print(f'[cyan]{worker_idx}: Done!')
    print()


def main(input_root='/path/to/ParallelDomain-4D/data',
         output_root='/path/to/ParallelDomain-4D/pcl',
         gpus='0,1,2,3', start_idx=0, end_idx=1600, debug=0, ignore_if_exist=1,
         sel_views='all', sel_frames='all', sel_modals='all'):

    default_views = 19
    default_frames = 50

    if isinstance(gpus, str):
        gpus = [int(x.strip()) for x in gpus.split(',')]
    elif isinstance(gpus, int):
        gpus = [gpus]

    if sel_views == 'all':
        sel_views = list(range(default_views))
    elif isinstance(sel_views, str):
        sel_views = [int(x.strip()) for x in sel_views.split(',')]
    if sel_frames == 'all':
        sel_frames = list(range(default_frames))
    elif isinstance(sel_frames, str):
        sel_frames = [int(x.strip()) for x in sel_frames.split(',')]
    if sel_modals == 'all':
        sel_modals = ['rgb', 'segm']  # , 'inst']
    elif isinstance(sel_modals, str):
        sel_modals = [x.strip() for x in sel_modals.split(',')]

    example_dns = os.listdir(input_root)
    example_dns = sorted([x for x in example_dns if 'scene' in x])
    example_dns = example_dns[start_idx:end_idx]
    example_pairs = [(os.path.join(input_root, x),
                      os.path.join(output_root, x)) for x in example_dns]

    if ignore_if_exist:
        filtered_examples = [(x, y) for (x, y) in example_pairs if not (is_finished(y, 40))]
    else:
        filtered_examples = example_pairs
    print(f'[cyan]Found {len(example_pairs)} examples to process, filtered down to '
          f'{len(filtered_examples)}...')

    worker_args_list = []
    for (worker_idx, gpu_idx) in enumerate(gpus):
        my_examples = filtered_examples[worker_idx::len(gpus)]
        worker_args_list.append((worker_idx, gpu_idx, len(gpus), my_examples,
                                 sel_views, sel_frames, sel_modals, debug, ignore_if_exist))

    print(f'[cyan]Splitting {len(filtered_examples)} examples across {len(gpus)} workers '
          f'with specified GPU devices: {gpus}...')
    start_time = time.time()

    if len(gpus) > 1:
        print(f'[cyan]Starting {len(gpus)} processes...')

        # https://github.com/pytorch/pytorch/issues/40403
        import torch
        import torch.multiprocessing
        torch.multiprocessing.set_start_method('spawn')

        with mp.Pool(processes=len(gpus)) as pool:
            results = pool.starmap(worker_fn, worker_args_list)

    else:
        print(f'[cyan]Calling method directly...')
        results = []
        for worker_args in worker_args_list:
            results.append(worker_fn(*worker_args))

    print(f'[magenta]Everything took {time.time() - start_time:.2f}s')

    print()
    print(f'[cyan]Done!')
    print()


if __name__ == '__main__':

    fire.Fire(main)

    pass
