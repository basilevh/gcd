'''
Created by Basile Van Hoorick for GCD, 2024.
Shared code for Kubric-4D generation and point cloud conversion scripts.
'''

# Library imports.
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def update_os_cpu_affinity(rank, max_world_size):
    # Ensure multiple of max_world_size (rounding down) to make numbers nicer (if needed).
    cpu_count = os.cpu_count()
    cpu_count = (cpu_count // max_world_size) * max_world_size

    if rank >= 0:
        rank = rank % max_world_size
        start = (rank * cpu_count) // max_world_size
        end = ((rank + 1) * cpu_count) // max_world_size
        affinity = set(range(start, end))
    else:
        affinity = set(range(0, cpu_count))

    try:
        os.sched_setaffinity(0, affinity)
    except:
        affinity = set(range(0, cpu_count))
        os.sched_setaffinity(0, affinity)

    print(f'{rank}: New CPU affinity: {os.sched_getaffinity(0)}')


def depth_to_rgb_vis(depth, max_depth=None):
    '''
    :depth (*, 1) array of float32.
    :return rgb_vis (*, 3) array of uint8.
    '''
    min_depth = 0.0
    if max_depth is None:
        max_depth = max(np.max(depth), 1e-6)

    depth = depth.copy()
    depth = np.clip(depth, 0.0, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)

    rgb_vis = plt.cm.viridis(2.0 / (depth + 1.0) - 1.0)[..., 0:3]
    rgb_vis = (rgb_vis * 255.0).astype(np.uint8)

    return rgb_vis


def segm_ids_to_rgb(segm_ids, hue_step=0.1, zero_black=True):
    '''
    :param segm_ids: (*) array of int.
    :return segm_rgb: (*, 3) array of uint8.
    '''
    if zero_black:
        hue = ((segm_ids - 1) * hue_step) % 1.0
    else:
        hue = (segm_ids * hue_step) % 1.0
    hue = (hue + 1.0) % 1.0

    segm_rgb = plt.cm.hsv(hue)[..., 0:3]
    if zero_black:
        segm_rgb[segm_ids == 0] = 0.0
    segm_rgb = (segm_rgb * 255.0).astype(np.uint8)

    return segm_rgb


def save_video(dst_fp, frames, fps, quality):
    print(f'[yellow]Saving video to: {dst_fp}')
    if isinstance(frames, list):
        frames = np.stack(frames)
    if frames.dtype.kind == 'f':
        frames = (frames * 255.0).astype(np.uint8)
    imageio.mimwrite(dst_fp, frames, format='ffmpeg', fps=float(fps),
                     macro_block_size=8, quality=int(quality))


def load_json(fp):
    with open(fp, 'r') as f:
        data = json.load(f)
    return data


def camera_to_world(xyz_camera, extrinsics):
    xyz_world = xyz_camera @ extrinsics[0:3, 0:3].T
    xyz_world += extrinsics[0:3, 3]
    return xyz_world


def world_to_camera(xyz_world, extrinsics):
    xyz_camera = xyz_world - extrinsics[0:3, 3]
    xyz_camera = xyz_camera @ extrinsics[0:3, 0:3]
    return xyz_camera


def unproject_pixels_to_points(rgb, depth, K, RT, process_dtype):
    '''
    This method supports features of arbitrary dimensionality, not just 3-channel pixels.
    :param rgb: (H, W, C) tensor of float.
    :param depth: (H, W) tensor of float.
    :param K: (3, 3) tensor of float.
    :param RT: (4, 4) tensor of float.
    :return xyzrgb: (N, 3 + C) tensor of float, where N = H * W.
    '''
    import torch
    (H, W, C) = rgb.shape

    # Create meshgrid for pixel coordinates.
    u_coord = torch.arange(W, dtype=process_dtype, device=rgb.device)  # (W).
    v_coord = torch.arange(H, dtype=process_dtype, device=rgb.device)  # (H).
    (u, v) = torch.meshgrid(u_coord, v_coord, indexing='xy')
    # 2x (H, W); first horizontal then vertical coords.

    # Unproject (u, v, depth) to (x, y, z) in camera coordinates.
    z = depth.view(-1)  # (H * W).
    u = u.reshape(-1)  # (H * W); horizontal coords; minor ordered values.
    v = v.reshape(-1)  # (H * W); vertical coords; major ordered values.
    x = (u - K[0, 2]) * z / K[0, 0]  # (H * W).
    y = (v - K[1, 2]) * z / K[1, 1]  # (H * W).

    # Stack to get (N, 3) points in camera coordinates.
    xyz_camera = torch.stack((x, y, z), dim=1)  # (H * W, 3).

    # Transform to world coordinates.
    xyz_world = camera_to_world(xyz_camera, RT)

    # Concatenate with RGB values.
    rgb_flat = rgb.view(-1, C)  # (H * W, C).
    xyzrgb = torch.cat((xyz_world, rgb_flat), dim=1)  # (H * W, 3 + C).

    return xyzrgb
