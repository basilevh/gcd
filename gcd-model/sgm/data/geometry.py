'''
Created by Basile Van Hoorick for GCD, 2024.
Fast, optimized code for "rendering" merged point clouds into pseudo-ground truth video pairs.
'''

# Library imports.
import cv2
import glob
import lovely_tensors
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import os
import time
import torch
import torchvision
import torchvision.transforms
from einops import rearrange
from lovely_numpy import lo
from pyquaternion import Quaternion
from rich import print

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)


def get_kubric_camera_matrices_torch(metadata):
    '''
    Adapted from convert_pcl_kubric.py.
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

        extrinsics_matrix = torch.eye(4, dtype=torch.float32)
        extrinsics_matrix[0:3, 0:3] = torch.tensor(rot_m)
        extrinsics_matrix[0:3, 3] = torch.tensor(rot_t)

        # Flip Y and Z axes of camera pose (so negate columns);
        # other stuff remains intact.
        extrinsics_matrix[0:3, 1] *= -1.0
        extrinsics_matrix[0:3, 2] *= -1.0

        # This is normalized (so sensor width and height are 1);
        # multiply by image resolution to get image coordinates.
        intrinsics_matrix = np.abs(metadata['camera']['K'])
        intrinsics_matrix = torch.tensor(intrinsics_matrix, dtype=torch.float32)

        all_extrinsics.append(extrinsics_matrix)
        all_intrinsics.append(intrinsics_matrix)

    all_extrinsics = torch.stack(all_extrinsics, dim=0)  # (T, 4, 4) tensor of float.
    all_intrinsics = torch.stack(all_intrinsics, dim=0)  # (T, 3, 3) tensor of float.

    return (all_intrinsics, all_extrinsics)


def get_pardom_intrinsics_matrix(intrinsics_dict):
    '''
    Adapted from convert_pcl_pardom.py.
    '''
    intrinsics_matrix = torch.tensor(
        [[intrinsics_dict['fx'], 0.0, intrinsics_dict['cx']],
         [0.0, intrinsics_dict['fy'], intrinsics_dict['cy']],
         [0.0, 0.0, 1.0]], dtype=torch.float32)
    return intrinsics_matrix


def get_pardom_extrinsics_matrix(extrinsics_dict):
    '''
    Adapted from convert_pcl_pardom.py.
    '''
    rot_q = extrinsics_dict['rotation']
    rot_t = extrinsics_dict['translation']
    rot_m = Quaternion(rot_q['qw'], rot_q['qx'], rot_q['qy'], rot_q['qz']).rotation_matrix
    extrinsics_matrix = torch.eye(4, dtype=torch.float32)
    extrinsics_matrix[0:3, 0:3] = torch.tensor(rot_m)
    extrinsics_matrix[0:3, 3] = torch.tensor([rot_t['x'], rot_t['y'], rot_t['z']])
    return extrinsics_matrix


def get_pardom_camera_matrices_torch(calibration):
    '''
    Adapted from convert_pcl_pardom.py.
    :return all_views: List of str with view names corresponding to camera matrix ordering.
    :return all_intrinsics: (V, 3, 3) tensor of float32.
    :return all_extrinsics: (V, 4, 4) tensor of float32.
    '''
    # NOTE: Camera parameters do not vary over time in this dataset.
    view_names = []
    all_intrinsics = dict()  # Maps view_name to (3, 3) tensor of float.
    all_extrinsics = dict()  # Maps view_name to (4, 4) tensor of float.

    for (view_name, intrinsics_dict, extrinsics_dict) in zip(
            calibration['names'], calibration['intrinsics'], calibration['extrinsics']):

        if 'velodyne' in view_name.lower():
            continue

        # NOTE: Unlike Kubric, the intrinsics matrix is already unnormalized (pixel coordinates).
        intrinsics_matrix = get_pardom_intrinsics_matrix(intrinsics_dict)  # (3, 3) tensor of float.
        extrinsics_matrix = get_pardom_extrinsics_matrix(extrinsics_dict)  # (4, 4) tensor of float.
        all_intrinsics[view_name] = intrinsics_matrix
        all_extrinsics[view_name] = extrinsics_matrix
        view_names.append(view_name)

    view_names = sorted(view_names)  # (V) list of str.
    all_intrinsics = torch.stack([all_intrinsics[view_name] for view_name in view_names], dim=0)
    # (V, 3, 3) tensor of float.
    all_extrinsics = torch.stack([all_extrinsics[view_name] for view_name in view_names], dim=0)
    # (V, 4, 4) tensor of float.

    # For reference: view_names =
    # ['camera0', 'camera1', 'camera10', 'camera11', 'camera12', 'camera13', 'camera14', 'camera15',
    # 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9',
    # 'yaw-0', 'yaw-60', 'yaw-neg-60']

    return (view_names, all_intrinsics, all_extrinsics)


def cartesian_from_spherical(spherical, deg2rad=False):
    '''
    :param spherical: (..., 3) array of float.
    :return cartesian: (..., 3) array of float.
    '''
    azimuth = spherical[..., 0]
    elevation = spherical[..., 1]
    radius = spherical[..., 2]
    if deg2rad:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    cartesian = np.stack([x, y, z], axis=-1)
    return cartesian


def spherical_from_cartesian(cartesian, rad2deg=False):
    '''
    :param cartesian: (..., 3) array of float.
    :return spherical: (..., 3) array of float.
    '''
    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]
    radius = np.linalg.norm(cartesian, ord=2, axis=-1)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.linalg.norm(cartesian[..., 0:2], ord=2, axis=-1))
    if rad2deg:
        azimuth = np.rad2deg(azimuth)
        elevation = np.rad2deg(elevation)
    spherical = np.stack([azimuth, elevation, radius], axis=-1)
    return spherical


def interpolate_spherical(cart_start, cart_end, alpha):
    '''
    :param cart_start: (3) array of float.
    :param cart_end: (3) array of float.
    :param alpha: single float.
    :return cart_interp: (3) array of float.
    '''
    spher_start = spherical_from_cartesian(cart_start)
    spher_end = spherical_from_cartesian(cart_end)
    if spher_end[0] - spher_start[0] > np.pi:
        spher_end[0] -= 2 * np.pi
    if spher_end[0] - spher_start[0] < -np.pi:
        spher_end[0] += 2 * np.pi
    if spher_end[1] - spher_start[1] > np.pi:
        spher_end[1] -= 2 * np.pi
    if spher_end[1] - spher_start[1] < -np.pi:
        spher_end[1] += 2 * np.pi
    spher_interp = spher_start * (1 - alpha) + spher_end * alpha
    cart_interp = cartesian_from_spherical(spher_interp)
    return cart_interp


def extrinsics_from_look_at(camera_position, camera_look_at):
    '''
    :param camera_position: (3) array of float.
    :param camera_look_at: (3) array of float.
    :return RT: (4, 4) array of float.
    '''
    # NOTE: In my convention (including Kubric and ParallelDomain),
    # the columns (= camera XYZ axes) should be: right, down, forward.

    # Calculate forward vector: Z.
    forward = (camera_look_at - camera_position)
    forward /= np.linalg.norm(forward)
    # Assume world's down vector: Y.
    world_down = np.array([0, 0, -1])
    # Calculate right vector: X = Y cross Z.
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    # Calculate actual down vector: Y = Z cross X.
    down = np.cross(forward, right)

    # Construct 4x4 extrinsics matrix.
    RT = np.eye(4)
    RT[0:3, 0:3] = np.stack([right, down, forward], axis=1)
    RT[0:3, 3] = camera_position

    return RT


def camera_to_world(xyz_camera, extrinsics):
    '''
    :param xyz_camera: (..., 3) array of float.
    :param extrinsics: (4, 4) array of float.
    :return xyz_world: (..., 3) array of float.
    '''
    xyz_world = xyz_camera @ extrinsics[0:3, 0:3].T
    xyz_world += extrinsics[0:3, 3]
    return xyz_world


def world_to_camera(xyz_world, extrinsics):
    '''
    :param xyz_world: (..., 3) array of float.
    :param extrinsics: (4, 4) array of float.
    :return xyz_camera: (..., 3) array of float.
    '''
    xyz_camera = xyz_world - extrinsics[0:3, 3]
    xyz_camera = xyz_camera @ extrinsics[0:3, 0:3]
    return xyz_camera


def project_points_to_pixels(xyzrgb, K, RT, H, W, spread_radius=2):
    '''
    NOTE: Keep in mind this is much faster on GPU!
    :param xyzrgb: (N, 6) tensor of float, where N can be any value.
    :param K: (3, 3) tensor of float.
    :param RT: (4, 4) tensor of float.
    :return img_norm: (H, W, 3) tensor of float.
    :return pixel_weights: (H, W, 1) tensor of float.
    :return uv: (N, 2) tensor of float.
    :return depth: (N, 1) tensor of float.
    '''
    # NOTE: Various del statements are used to free up VRAM as soon as possible. These optimizations
    # save around 40% of VRAM usage. Also, the VRAM is proportional to the total number of workers
    # (across all processes). See also: https://pytorch.org/docs/stable/torch_cuda_memory.html

    xyzrgb = xyzrgb.type(torch.float64)
    K = K.type(torch.float64).to(xyzrgb.device)
    RT = RT.type(torch.float64).to(xyzrgb.device)

    # Extracting xyz and projecting to camera coordinates.
    xyz_world = xyzrgb[:, 0:3]  # (N, 3).
    xyz_camera = world_to_camera(xyz_world, RT)
    del xyz_world

    # Projecting to pixel coordinates.
    uv = torch.mm(K, xyz_camera.T).T  # (N, 3).
    uv = uv[:, 0:2] / uv[:, 2:3]  # Divide by z to get image coordinates.

    # Convert to integer pixel coordinates and apply mask.
    uv_int = (uv + 0.5).type(torch.int32)  # (N, 2) with (horizontal, vertical) coordinates.
    depth = xyz_camera[:, 2:3]  # Depth is z in camera coordinates.
    mask = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & \
           (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) & \
           (depth[:, 0] > 0.1)
    del xyz_camera

    # Filter points that are inside the image and have valid depth.
    rgb_filter = xyzrgb[mask][:, 3:6]  # (M, 3).
    uv_int_filter = uv_int[mask]  # (M, 2).
    depth_filter = depth[mask]  # (M, 1).
    del mask

    # Convert 2D indices to 1D. These coordinates are horizontal minor and vertical major.
    inds_flat = uv_int_filter[:, 1] * W + uv_int_filter[:, 0]
    del uv_int_filter

    # Make sure closer points are considered significantly more important in this aggregation.
    if depth_filter.max() >= 64.0:
        # We're probably dealing with ParallelDomain which has very far away points.
        strength = 256.0
        depth_filter = torch.sqrt(depth_filter)
        depth_filter = torch.clamp(depth_filter, 0.0, 32.0)
    else:
        # We're probably dealing with Kubric.
        strength = 512.0
        # NOTE: In Kubric, strength 128 or above creates overflow in float32,
        # and 1024 or above creates overflow in float64.

    depth_norm = depth_filter / depth_filter.max() * 2.0 - 1.0  # (M, 1) with values in [-1, 1].
    del depth_filter
    point_weights = torch.exp(-depth_norm * strength)  # (M, 1) and decreasing with depth.
    del depth_norm
    weighted_rgb = rgb_filter * point_weights  # (M, 3) and gets darker with depth.
    del rgb_filter

    # Normalize by accumulating weighted point-to-pixel counts (becomes denominator).
    pixel_weights_flat = torch.zeros(H * W, 1, dtype=torch.float64, device=xyzrgb.device)
    spreaded_index_add(pixel_weights_flat, inds_flat, point_weights, H, W, spread_radius)
    del point_weights
    # (H * W * 1).

    # Accumulate weighted color pixel values themselves (becomes numerator).
    img_flat = torch.zeros(H * W, 3, dtype=torch.float64, device=xyzrgb.device)
    spreaded_index_add(img_flat, inds_flat, weighted_rgb, H, W, spread_radius)
    del weighted_rgb
    # (H * W * 3).

    # Avoid division by zero, but make it clear exactly where no point contributed to a pixel.
    pixel_weights = pixel_weights_flat.view(H, W, 1)  # Reshape to (H, W, 1).
    pixel_weights[pixel_weights <= 0.0] = -1.0

    # Also calculate direct counts for debugging.
    if 0:
        count_flat = torch.zeros(H * W, 1, dtype=torch.int32, device=xyzrgb.device)  # (H * W * 1).
        ones = torch.ones_like(point_weights, dtype=torch.int32)
        count_flat.index_add_(0, inds_flat, ones)
        count = count_flat.view(H, W, 1)  # Reshape back to (H, W, 1).
        count = torch.clamp(count, min=1)  # Avoid division by zero.

    # Normalize and clip final pixel values.
    img = img_flat.view(H, W, 3)  # Reshape to (H, W, 3).
    img_norm = img / pixel_weights
    img_norm = torch.clamp(img_norm, 0.0, 1.0)
    img_norm = img_norm.type(torch.float32)

    return (img_norm, pixel_weights, uv, depth)


def spreaded_index_add(tensor, indices, values, H, W, radius):
    '''
    :param tensor: (N, C) tensor of any type.
    :param indices: (M) tensor of int with values in [0, N - 1].
    :param values: (M, C) tensor of any type.
    :param H, W, radius (int): Image dimensions and spread radius.
    :return tensor: (N, C) tensor of any type.
    '''
    # Benchmark stuff for debugging.
    if 0:
        NI = 10

        t = tensor.detach().clone().cpu()
        i = indices.detach().clone().cpu()
        v = values.detach().clone().cpu()
        start_time = time.time()
        for _ in range(NI):
            t.index_add_(0, i, v)
        print(f'[magenta]torch.index_add_ on CPU takes '
              f'~{(time.time() - start_time) * 1000.0 / NI:.1f}ms')

        t = tensor.detach().clone().cuda(1)
        i = indices.detach().clone().cuda(1)
        v = values.detach().clone().cuda(1)
        start_time = time.time()
        for _ in range(NI):
            t.index_add_(0, i, v)
        print(f'[magenta]torch.index_add_ on GPU takes '
              f'~{(time.time() - start_time) * 1000.0 / NI:.1f}ms')

    # Accumulate values at indices in-place within tensor.
    tensor.index_add_(0, indices, values)

    # NOTE: Only the above line would be the default / vanilla operation, but we wish to
    # avoid random pixel holes inbetween points, which requires a more advanced algorithm.
    left = radius // 2
    right = (radius + 1) // 2
    offset_list = []
    for dx in range(-left, right + 1):
        for dy in range(-left, right + 1):
            if dx == 0 and dy == 0:
                continue
            offset_list.append((dx, dy))
    # when radius = 1: [(1, 0), (0, 1), (1, 1)].
    # when radius = 2: [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)].
    # when radius = 3: x and y go from -1 to 2 inclusive, etc..

    for (dx, dy) in offset_list:
        # Spread values to neighboring pixels.
        inds_x = indices % W + dx
        inds_y = indices // W + dy
        shift_inds = inds_y * W + inds_x

        # Also avoid leaking across image borders.
        # shift_inds = torch.clamp(shift_inds, min=0, max=H * W - 1)
        mask = (inds_x >= 0) & (inds_x < W) & (inds_y >= 0) & (inds_y < H)
        mask_inds = shift_inds[mask]
        mask_values = values[mask] * 0.02  # Weaken as original pixels should have priority.

        tensor.index_add_(0, mask_inds, mask_values)

    return tensor


def blur_into_black(img, kernel_size=5, sigma=1.5):
    '''
    :param img: (H, W, 3) tensor of float.
    :param radius (int): Blur radius.
    '''
    black_mask = (img.sum(dim=-1) == 0.0)[None]  # (1, H, W).
    img = rearrange(img, 'h w c -> c h w')  # (3, H, W).

    # First leak valid content into invalid regions.
    img2 = gaussian_blur_masked_vectorized(img, ~black_mask, black_mask, kernel_size, sigma)

    # Then apply slight, gentle blurring to smooth the rough edges due to both spreaded_index_add
    # and previous operation.
    img2 = torchvision.transforms.functional.gaussian_blur(
        img2, kernel_size=3, sigma=0.6)

    img2 = rearrange(img2, 'c h w -> h w c')  # (H, W, 3).
    return img2


def gaussian_blur_masked_vectorized(img, borrow_mask, apply_mask, kernel_size, sigma):
    '''
    Apply Gaussian blur to an image but only considering pixels within borrow_mask for the
        convolution content, and change only pixels within apply_mask.
    :param img: (C, H, W) tensor of float.
    :param borrow_mask: (1, H, W) tensor of bool indicating which pixels to use.
    :param apply_mask: (1, H, W) tensor of bool indicating which pixels to modify.
    '''
    borrow_mask = borrow_mask.type(torch.float64)

    # in 2D:
    blur_img = torchvision.transforms.functional.gaussian_blur(
        img, kernel_size=kernel_size, sigma=sigma)
    blur_mask = torchvision.transforms.functional.gaussian_blur(
        borrow_mask, kernel_size=kernel_size, sigma=sigma)

    blur_mask = blur_mask.clamp(min=1e-7)
    leak_img = blur_img / blur_mask

    final_img = img * (~apply_mask) + leak_img * apply_mask
    return final_img
