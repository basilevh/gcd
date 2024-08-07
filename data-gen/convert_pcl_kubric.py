'''
Created by Basile Van Hoorick for GCD, 2024.
Computes & caches merged point clouds for efficient Kubric-4D data loading.
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


def is_finished(output_dp, expect_files):
    exist_files = len(glob.glob(os.path.join(output_dp, '*.pt')))
    return exist_files >= expect_files


def get_kubric_camera_matrices_torch(metadata):
    '''
    :return all_intrinsics: (T, 3, 3) tensor of float32.
    :return all_extrinsics: (T, 4, 4) tensor of float32.
    '''
    import torch

    T = metadata['scene']['num_frames']
    all_extrinsics = []  # List of (4, 4) tensor of float.
    all_intrinsics = []  # List of (3, 3) tensor of float.

    for t in range(T):
        rot_q = np.array(metadata['camera']['quaternions'][t])
        rot_t = np.array(metadata['camera']['positions'][t])
        rot_m = Quaternion(rot_q).rotation_matrix

        extrinsics_matrix = torch.eye(4, dtype=process_dtype)
        extrinsics_matrix[0:3, 0:3] = torch.tensor(rot_m)
        extrinsics_matrix[0:3, 3] = torch.tensor(rot_t)

        # Flip Y and Z axes of camera pose (so negate columns);
        # other stuff remains intact.
        extrinsics_matrix[0:3, 1] *= -1.0
        extrinsics_matrix[0:3, 2] *= -1.0

        # This is normalized (so sensor width and height are 1);
        # multiply by image resolution to get image coordinates.
        intrinsics_matrix = np.abs(metadata['camera']['K'])
        intrinsics_matrix = torch.tensor(intrinsics_matrix, dtype=process_dtype)

        all_extrinsics.append(extrinsics_matrix)
        all_intrinsics.append(intrinsics_matrix)

    all_extrinsics = torch.stack(all_extrinsics, dim=0)  # (T, 4, 4) tensor of float.
    all_intrinsics = torch.stack(all_intrinsics, dim=0)  # (T, 3, 3) tensor of float.

    return (all_intrinsics, all_extrinsics)


def correct_depth_ball_plane_torch(depth_ball, intrinsics):
    '''
    Correct the depth values from the camera point to the camera plane using PyTorch.
    NOTE: Verified to be equivalent to correct_depth_ball_plane_numpy().
    :param depth_ball: (*, H, W) tensor of float.
    :param intrinsics: (3, 3) tensor of float.
    :return depth_plane: (*, H, W) tensor of float.
    '''
    import torch
    (H, W) = depth_ball.shape[-2:]

    fov_x = 2.0 * torch.atan(W / (2.0 * torch.abs(intrinsics[0, 0])))
    fov_y = 2.0 * torch.atan(H / (2.0 * torch.abs(intrinsics[1, 1])))

    angles_x = torch.linspace(-fov_x / 2.0, fov_x / 2.0, W,
                              dtype=process_dtype, device=depth_ball.device)
    angles_y = torch.linspace(-fov_y / 2.0, fov_y / 2.0, H,
                              dtype=process_dtype, device=depth_ball.device)

    mismatch_x = torch.tan(angles_x)
    mismatch_y = torch.tan(angles_y)

    correction = torch.sqrt(mismatch_x[None, :] ** 2 + mismatch_y[:, None] ** 2 + 1.0)
    depth_plane = depth_ball / correction

    return depth_plane


def process_example(worker_idx, gpu_idx, example,
                    sel_views, sel_frames, sel_modals, debug, ignore_if_exist):
    import torch

    (scene_dp, output_dp) = example

    expect_files = len(sel_frames) * 1  # * len(sel_modals)
    if ignore_if_exist and is_finished(output_dp, expect_files):
        print(f'[yellow]{worker_idx}: Skipping {output_dp} because it already contains '
              f'no less than {expect_files} files...')
        return False

    device = torch.device(f'cuda:{gpu_idx}')

    # NOTE: To save memory, we do not retain large arrays or tensors in memory over time!
    print()
    print(f'[yellow]{worker_idx}: Processing scene {example}...')
    print(f'[yellow]{worker_idx}: Unprojecting all pixels to points...')
    print(f'[yellow]{worker_idx}: Saving merged point clouds to disk...')
    start_time = time.time()

    scene_dn = os.path.basename(scene_dp)

    # These arrays are constructed as lists over viewpoints.
    all_intrinsics = []  # (T, V, 3, 3) tensor of float.
    all_extrinsics = []  # (T, V, 4, 4) tensor of float.

    for (j, v) in enumerate(sel_views):
        metadata_fp = os.path.join(scene_dp, f'{scene_dn}_p0_v{v}.json')
        metadata = data_utils.load_json(metadata_fp)

        (camera_K, camera_R) = get_kubric_camera_matrices_torch(metadata)
        intrinsics_matrix = camera_K[sel_frames].to(device)  # (T, 3, 3) tensor of float.
        extrinsics_matrix = camera_R[sel_frames].to(device)  # (T, 4, 4) tensor of float.

        # NOTE: In Kubric, the instrinsics are with respect to normalized image space [0, 1]^2.
        (W, H) = metadata['scene']['resolution']
        intrinsics_matrix[..., 0, :] *= W
        intrinsics_matrix[..., 1, :] *= H

        all_intrinsics.append(intrinsics_matrix)
        all_extrinsics.append(extrinsics_matrix)

    all_intrinsics = torch.stack(all_intrinsics, dim=1)  # (T, V, 3, 3) tensor of float.
    all_extrinsics = torch.stack(all_extrinsics, dim=1)  # (T, V, 4, 4) tensor of float.

    for (i, t) in tqdm.tqdm(enumerate(sel_frames)):

        # For this frame, load data from all viewpoints.
        frame_rgb = []
        frame_depth = []
        frame_instance_rgb = []

        for (j, v) in enumerate(sel_views):

            frames_dp = os.path.join(scene_dp, f'frames_p0_v{j}')
            rgb_fp = os.path.join(frames_dp, f'rgba_{t:05d}.png')
            depth_fp = os.path.join(frames_dp, f'depth_{t:05d}.tiff')
            instance_rgb_fp = os.path.join(frames_dp, f'segmentation_{t:05d}.png')

            rgb = plt.imread(rgb_fp)[..., 0:3]
            rgb = torch.tensor(rgb, dtype=torch.float32, device=device)
            # (H, W, 3) tensor of float.

            depth_ball = np.array(PIL.Image.open(depth_fp))
            depth_ball = torch.tensor(depth_ball, dtype=torch.float32, device=device)
            # (H, W) tensor of float.

            # Immediately apply correction to depth.
            depth_plane = correct_depth_ball_plane_torch(depth_ball, all_intrinsics[i, j])
            # (H, W) tensor of float.

            instance_rgb = plt.imread(instance_rgb_fp)[..., 0:3]
            instance_rgb = torch.tensor(instance_rgb, dtype=torch.float32, device=device)
            # (H, W, 3) tensor of float.

            frame_rgb.append(rgb)
            frame_depth.append(depth_plane)
            frame_instance_rgb.append(instance_rgb)

        frame_rgb = torch.stack(frame_rgb, dim=0)  # (V, H, W, 3) tensor of float.
        frame_depth = torch.stack(frame_depth, dim=0)  # (V, H, W) tensor of float.
        frame_instance_rgb = torch.stack(frame_instance_rgb, dim=0)  # (V, H, W, 3) tensor of float.

        # For this frame, deproject & merge point clouds.
        frame_xyzfeats = []

        for (j, v) in enumerate(sel_views):
            assert 'rgb' in sel_modals
            assert 'segm' in sel_modals

            rgb = frame_rgb[j]  # (H, W, 3) tensor of float.
            segm = frame_instance_rgb[j]  # (H, W, 3) tensor of float.
            feats = torch.cat([rgb, segm], dim=-1)  # (H, W, 6) tensor of float.

            depth = frame_depth[j]  # (H, W) tensor of float.
            K = all_intrinsics[i, j]  # (3, 3) tensor of float.
            RT = all_extrinsics[i, j]  # (4, 4) tensor of float.

            xyzfeats = data_utils.unproject_pixels_to_points(feats, depth, K, RT, process_dtype)
            frame_xyzfeats.append(xyzfeats)
            # (N, 3 + C) tensor of float.

        frame_xyzfeats = torch.stack(frame_xyzfeats, dim=0)
        # (V, N, 3 + C) tensor of float.

        # Now compress stuff to save disk space.
        # See https://pytorch.org/docs/stable/tensors.html#id4 -- 10 significand bits.
        # NOTE: I verified these values and it creates a relative error of 0.1% on average.
        # The error is highest (3%) for very small numbers (abs < 1e-5).
        store_xyz = frame_xyzfeats[..., 0:3].clone().type(store_xyz_dtype)
        store_rgb = (frame_xyzfeats[..., 3:6].clone() * 255.0).type(torch.uint8)
        store_segm = (frame_xyzfeats[..., 6:9].clone() * 255.0).type(torch.uint8)
        store_xyz = store_xyz.detach().cpu()
        store_rgb = store_rgb.detach().cpu()
        store_segm = store_segm.detach().cpu()

        # For this frame, save results to disk.
        os.makedirs(output_dp, exist_ok=True)
        modals_fn = '_'.join(sel_modals)
        dst_fp = os.path.join(output_dp, f'pcl_{modals_fn}_{t:05d}.pt')
        torch.save([store_xyz, store_rgb, store_segm], dst_fp)

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
        process_example(worker_idx, gpu_idx, example,
                        sel_views, sel_frames, sel_modals, debug, ignore_if_exist)

    print()
    print(f'[cyan]{worker_idx}: Done!')
    print()


def main(input_root='/path/to/Kubric-4D/data',
         output_root='/path/to/Kubric-4D/pcl',
         gpus='0,1,2,3', start_idx=0, end_idx=3000, debug=0, ignore_if_exist=1,
         sel_views='all', sel_frames='all', sel_modals='all'):

    default_views = 16
    default_frames = 60

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
        sel_modals = ['rgb', 'segm']
    elif isinstance(sel_modals, str):
        sel_modals = [x.strip() for x in sel_modals.split(',')]

    example_dns = os.listdir(input_root)
    example_dns = sorted([x for x in example_dns if 'scn' in x])
    example_dns = example_dns[start_idx:end_idx]
    example_pairs = [(os.path.join(input_root, x),
                      os.path.join(output_root, x)) for x in example_dns]

    if ignore_if_exist:
        expect_files = len(sel_frames) * 1  # * len(sel_modals)
        filtered_examples = [(x, y) for (x, y) in example_pairs
                             if not (is_finished(y, expect_files))]
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
