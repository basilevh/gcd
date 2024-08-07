'''
Created by Basile Van Hoorick for GCD, 2024.
'''

# Library imports.
import cv2
import glob
import json
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import os
from einops import rearrange
from lovely_numpy import lo
from pyquaternion import Quaternion
from rich import print

np.set_printoptions(precision=3, suppress=True)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
video_extensions = ['.mp4', '.avi', '.mov', '.flv', '.mkv', '.webm']


def resize_video(video_array, target_height, target_width):
    # Calculate aspect ratios
    original_height, original_width = video_array.shape[1:3]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Crop to correct aspect ratio
    if original_aspect > target_aspect:
        # Crop width
        new_width = int(original_height * target_aspect)
        start = (original_width - new_width) // 2
        cropped_video = video_array[:, :, start: start + new_width, :]
    else:
        # Crop height
        new_height = int(original_width / target_aspect)
        start = (original_height - new_height) // 2
        cropped_video = video_array[:, start: start + new_height, :, :]

    # Resize video
    resized_video = np.zeros(
        (video_array.shape[0], target_height, target_width, video_array.shape[3])
    )
    for i, frame in enumerate(cropped_video):
        resized_video[i] = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

    return resized_video


def center_crop_torch(image, aspect_ratio):
    '''
    :param image (..., H, W) array or tensor.
    :param aspect_ratio (float): Desired width / height.
    '''
    (H, W) = image.shape[-2:]
    video_ar = W / H
    do_crop = True

    if video_ar > aspect_ratio + 2e-3:
        # Crop width.
        crop_width = int(H * aspect_ratio)
        crop_height = H
    elif video_ar < aspect_ratio - 2e-3:
        # Crop height.
        crop_width = W
        crop_height = int(W / aspect_ratio)
    else:
        do_crop = False

    if do_crop:
        crop_y1 = (H - crop_height) // 2
        crop_y2 = crop_y1 + crop_height
        crop_x1 = (W - crop_width) // 2
        crop_x2 = crop_x1 + crop_width

        image = image[..., crop_y1:crop_y2, crop_x1:crop_x2]

    return image


def center_crop_numpy(image, aspect_ratio, warn_spatial):
    '''
    :param image (..., H, W, C) array or tensor.
    :param aspect_ratio (float): Desired width / height.
    '''
    (H, W) = image.shape[-3:-1]
    video_ar = W / H
    do_crop = True

    if video_ar > aspect_ratio + 2e-3:
        # Crop width.
        crop_width = int(H * aspect_ratio)
        crop_height = H
    elif video_ar < aspect_ratio - 2e-3:
        # Crop height.
        crop_width = W
        crop_height = int(W / aspect_ratio)
    else:
        do_crop = False

    if do_crop:
        if warn_spatial:
            print(f'[orange3]Warning: Cropping video from {W} x {H} '
                  f'to {crop_width} x {crop_height}.')
        crop_y1 = (H - crop_height) // 2
        crop_y2 = crop_y1 + crop_height
        crop_x1 = (W - crop_width) // 2
        crop_x2 = crop_x1 + crop_width

        image = image[..., crop_y1:crop_y2, crop_x1:crop_x2, :]

    return image


def load_rgb_image(src_fp, center_crop, frame_width, frame_height, warn_spatial):
    '''
    NOTE: This assumes RGB modality.
    :return rgb: (3, H, W) array of float32 in [-1, 1].
    '''
    rgb = plt.imread(src_fp)
    # (H, W, 4) array of float32 in [0, 1].
    rgb = process_image(rgb, center_crop, frame_width, frame_height, warn_spatial)
    # (3, H, W) array of float32 in [-1, 1].
    return rgb


def process_image(rgb, center_crop, frame_width, frame_height, warn_spatial):
    '''
    NOTE: This assumes RGB modality.
    :param rgb: (H, W, 3+) array of float32 in [0, 1].
    :return rgb: (3, H, W) array of float32 in [-1, 1].
    '''
    rgb = rgb[..., 0:3]

    if rgb.dtype.kind in ['i', 'u']:
        rgb = (rgb / 255.0).astype(np.float32)
    else:
        rgb = rgb.astype(np.float32)
    # (H, W, 3) array of float32 in [0, 1].

    if center_crop:
        rgb = center_crop_numpy(
            rgb, frame_width / frame_height, warn_spatial)

    if frame_width > 0 and frame_height > 0 and \
            (rgb.shape[1] != frame_width or rgb.shape[0] != frame_height):
        if warn_spatial:
            print(f'[orange3]Warning: Resizing image/frame from '
                  f'{rgb.shape[1]} x {rgb.shape[0]} to {frame_width} x {frame_height}.')
        rgb = cv2.resize(rgb, (frame_width, frame_height),
                         interpolation=cv2.INTER_LINEAR)

    rgb = rgb * 2.0 - 1.0
    rgb = rearrange(rgb, 'H W C -> C H W')
    # (3, H, W) array of float32 in [-1, 1].

    return rgb


def load_video_mp4(
        src_fp, clip_frames, center_crop, frame_width, frame_height, warn_spatial):
    '''
    NOTE: This assumes RGB modality.
    :return rgb: (Tc, 3, H, W) array of float32 in [-1, 1].
    '''
    rgb_raw = np.array(mediapy.read_video(src_fp))

    rgb = rgb_raw[clip_frames]
    rgb = rgb[..., 0:3]

    if rgb.dtype.kind in ['i', 'u']:
        rgb = (rgb / 255.0).astype(np.float32)
    else:
        rgb = rgb.astype(np.float32)
    # (Tc, 3, H, W) array of float32 in [0, 1].

    if center_crop:
        rgb = center_crop_numpy(
            rgb, frame_width / frame_height, warn_spatial)

    if frame_width > 0 and frame_height > 0 and \
            (rgb.shape[2] != frame_width or rgb.shape[1] != frame_height):
        if warn_spatial:
            print(f'[orange3]Warning: Resizing video from '
                  f'{rgb.shape[2]} x {rgb.shape[1]} to {frame_width} x {frame_height}.')
        rgb = np.stack([cv2.resize(x, (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR) for x in rgb], axis=0)

    rgb = rgb * 2.0 - 1.0
    rgb = rearrange(rgb, 'T H W C -> T C H W')
    # (Tc, 3, H, W) array of float32 in [-1, 1].

    return rgb


def load_kubric_video_rgb_frames(
        src_dps, clip_frames, center_crop, frame_width, frame_height, warn_spatial):
    '''
    NOTE: This assumes RGB modality.
    NOTE: This method supports rotating videos,
        by specifying a list of possibly different directories.
    :return rgb: (Tc, 3, H, W) array of float32 in [-1, 1].
    '''
    rgb = []

    if isinstance(src_dps, list):
        assert len(src_dps) == len(clip_frames)
    else:
        src_dps = [src_dps] * len(clip_frames)

    for (src_dp, t) in zip(src_dps, clip_frames):
        cur_fp = os.path.join(src_dp, f'rgba_{t:05d}.png')
        cur_rgb = load_rgb_image(cur_fp, center_crop, frame_width, frame_height,
                                 warn_spatial and t == 0)
        rgb.append(cur_rgb)
        # (3, H, W) array of float32 in [-1, 1].

    rgb = np.stack(rgb, axis=0)
    # (Tc, 3, H, W) array of float32 in [-1, 1].

    return rgb


def load_video_all_frames(
        src_dp, clip_frames, center_crop, frame_width, frame_height, warn_spatial):
    '''
    NOTE: This assumes RGB modality.
    :return rgb: (Tc, 3, H, W) array of float32 in [-1, 1].
    '''
    rgb = []

    src_fns = sorted(glob.glob(os.path.join(src_dp, '*.*')))
    src_fns = [fn for fn in src_fns if os.path.splitext(fn)[1].lower() in image_extensions]
    src_fps = [os.path.join(src_dp, fn) for fn in src_fns]
    src_fps = np.array(src_fps)[clip_frames]

    for (f, src_fp) in enumerate(src_fps):
        cur_rgb = load_rgb_image(src_fp, center_crop, frame_width, frame_height,
                                 warn_spatial and f == 0)
        rgb.append(cur_rgb)
        # (3, H, W) array of float32 in [-1, 1].

    rgb = np.stack(rgb, axis=0)
    # (Tc, 3, H, W) array of float32 in [-1, 1].

    return rgb


def get_pardom_camera_dn(ego_magic, view_idx):
    '''
    :param ego_magic (str).
    :param view_idx (int).
    '''
    if ego_magic == 'ego':
        # 0 - 2 = From left to right.
        camera_dn = ['yaw-60', 'yaw-0', 'yaw-neg-60'][view_idx]
    elif ego_magic == 'magic':
        # 0 - 15 = From back view counterclockwise.
        camera_dn = f'camera{view_idx}'
    else:
        raise ValueError(ego_magic)
    return camera_dn


def load_pardom_frame(scene_dp, modality, camera, time_idx):
    '''
    :param scene_dp (str).
    :param modality (str).
    :param camera (str).
    :param time_idx (int).
    :return frame: (H, W) array of float32 in [0, inf] if depth,
        or (H, W, 4) array of float32 in [0, 1] if RGB,
        or (H, W) array of uint32 in [0, 256^3 - 1] if segmentation.
    '''
    # in ParallelDomain-4D, we have:
    # 000000000000000005.png, 000000000000000015.png, ..., 000000000000000495.png.
    if 'depth' in modality:
        frame_fn = f'{time_idx * 10 + 5:018d}.npz'
    else:
        frame_fn = f'{time_idx * 10 + 5:018d}.png'
    frame_fp = os.path.join(scene_dp, modality, camera, frame_fn)

    # print(f'[yellow]Loading raw frame from {frame_fp}...')
    if 'depth' in modality:
        frame = np.load(frame_fp)['data']
        # (H, W) array of float32 in [0, inf).
    else:
        frame = plt.imread(frame_fp)
        # (H, W, 4) array of float32 in [0, 1].

    if 'segmentation' in modality:
        frame = (frame * 255.0).astype(np.int32)  # (H, W, 4) array of int32 in [0, 255].
        # NOTE: Only the first 3 channels appear to be used, and alpha is all 1.
        frame = frame[..., 0] + frame[..., 1] * 256 + frame[..., 2] * 256 * 256
        # (H, W) array of uint32 in [0, 16777215].

    return frame


def visualize_pardom_frame(frame, modality, camera, ontology):
    '''
    :param frame: Varies (see above).
    :param modality (str).
    :param camera (str).
    :param ontology (dict).
    :return vis_frame: (H, W, 3) array of float32 in [0, 1].
    '''
    mag_norm = 24.0

    if 'depth' in modality:
        # frame = (H, W) array of float32 in [0, inf).
        if 1:
            depth_vis = np.exp(-frame / 12.0)
            depth_min = np.min(depth_vis)
            depth_max = np.max(depth_vis)
            depth_vis = (depth_vis - depth_min) / (depth_max - depth_min + 1e-7)
            # (H, W) array of float32 in [0, 1].

        if 0:
            period = 16.0
            depth_vis = period - np.abs(frame % (period * 2.0) - period)
            depth_vis[frame >= period * 20.0] = 0.0
            depth_vis = depth_vis / period
            # (H, W) array of float32 in [0, 1].

        depth_rgb = plt.cm.plasma(depth_vis)[..., 0:3]
        vis_frame = depth_rgb
        # (H, W, 3) array of float32 in [0, 1].

    elif 'instance' in modality:
        # frame = (H, W) array of int32 in [0, 16777215].
        instance_id_rgb_map = ontology['instance_id_rgb_map']
        instance_rgb = instance_id_rgb_map[frame % 65536].numpy()

        vis_frame = instance_rgb
        # (H, W, 3) array of float32 in [0, 1].

    elif 'motion' in modality:
        # frame = (H, W, 4) array of float32 in [0, 1].
        dx = frame[..., 0] + frame[..., 1] * 256.0 - 128.0  # (H, W) array of float32 in [0, 256].
        dy = frame[..., 2] + frame[..., 3] * 256.0 - 128.0  # (H, W) array of float32 in [0, 256].
        angle = np.arctan2(dy, dx)  # (H, W) array of float32 in [-pi, pi].
        mag = np.sqrt(dx ** 2 + dy ** 2)  # (H, W) array of float32 in [0, 256].
        hue = (angle + np.pi) / (2.0 * np.pi)  # (H, W) array of float32 in [0, 1].
        # value = np.cbrt(mag / mag_norm)  # (H, W) array of float32 in [0, 1].
        value = np.sqrt(mag / (mag.max() + 1e-7))  # (H, W) array of float32 in [0, 1].
        value = np.clip(value, 0.0, 1.0)

        flow_hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
        # (H, W, 3) array of float32 in [0, 1].
        flow_rgb = matplotlib.colors.hsv_to_rgb(flow_hsv)
        # (H, W, 3) array of float32 in [0, 1].

        vis_frame = flow_rgb
        # (H, W, 3) array of float32 in [0, 1].

    elif 'rgb' in modality:
        # frame = (H, W, 4) array of float32 in [0, 1].
        vis_frame = frame[..., 0:3]  # (H, W, 3) array of float32 in [0, 1].

    elif 'semantic' in modality:
        # frame = (H, W) array of int32 in [0, 16777215].
        semantic_id_rgb_map = ontology['semantic_id_rgb_map']
        semantic_rgb = semantic_id_rgb_map[frame].numpy()

        vis_frame = semantic_rgb
        # (H, W, 3) array of float32 in [0, 1].

    elif 'surface' in modality:
        # frame = (H, W, 4) array of float32 in [0, 1].
        # NOTE: Only the first 3 channels appear to be used, and alpha is all 1.
        dx = frame[..., 0]
        dy = frame[..., 1]
        dz = frame[..., 2]

        vis_frame = frame[..., 0:3]
        # (H, W, 3) array of float32 in [0, 1].

    if (vis_frame < 0.0).any() or (vis_frame > 1.0).any():
        raise RuntimeError()

    return vis_frame


def load_pardom_video_vis_frames(scene_dp, modality, ego_magic, view_inds, ontology,
                                 clip_frames, center_crop, frame_width, frame_height):
    '''
    NOTE: This is not necessarily RGB modality -- automatic conversion will be done as needed.
    :return vis_frames: (Tcl, 3, Hp, Wp) array of float32 in [-1, 1].
    '''
    if modality == 'segm':
        modality = 'semantic_segmentation_2d'

    vis_frames = []

    if isinstance(view_inds, list):
        assert len(view_inds) == len(clip_frames)
    else:
        view_inds = [view_inds] * len(clip_frames)

    for (view_idx, frame_idx) in zip(view_inds, clip_frames):
        camera = get_pardom_camera_dn(ego_magic, view_idx)

        cur_frame = load_pardom_frame(scene_dp, modality, camera, frame_idx)
        # Varies (see method description).

        cur_vis = visualize_pardom_frame(cur_frame, modality, camera, ontology)
        # (Hp, Wp, 3) array of float32 in [0, 1].

        cur_vis = process_image(cur_vis, center_crop, frame_width, frame_height, False)
        # (3, Hp, Wp) array of float32 in [-1, 1].

        vis_frames.append(cur_vis)

    vis_frames = np.stack(vis_frames, axis=0)
    # (Tcl, 3, Hp, Wp) array of float32 in [-1, 1].

    return vis_frames


def load_json(fp):
    with open(fp, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fp):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, cls=JsonNumpyEncoder)


class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JsonNumpyEncoder, self).default(obj)


def construct_trajectory(spherical_start, spherical_end, trajectory, model_frames, move_time):
    '''
    :param spherical_start: (3,) array of float32.
    :param spherical_end: (3,) array of float32.
    :param trajectory (str).
    :param model_frames (int).
    :param move_time (int).
    :return spherical_src: (Tcm, 3) array of float32.
    :return spherical_dst: (Tcm, 3) array of float32.
    '''
    Tcm = model_frames

    # Determine input camera trajectory.
    spherical_src = np.tile(spherical_start[None], (Tcm, 1))
    # (Tcm, 3) array of float32.

    # Determine output camera trajectory.
    spherical_dst = np.tile(spherical_end[None], (Tcm, 1))
    if move_time >= 1:
        for t in range(0, move_time):
            if trajectory == 'interpol_linear':
                alpha = t / move_time
            elif trajectory == 'interpol_sine':
                alpha = (1.0 - np.cos(t / move_time * np.pi)) / 2.0
            else:
                raise ValueError(f'Unknown trajectory: {trajectory}')
            spherical_dst[t] = spherical_start * (1.0 - alpha) + spherical_end * alpha
    # (Tcm, 3) array of float32.

    return (spherical_src, spherical_dst)
