'''
Created by Basile Van Hoorick for GCD, 2024.
'''

# Library imports.
import cv2
import functools
import glob
import imageio
import json
import lovely_tensors
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import plotly.graph_objects as go
import sklearn
import sklearn.decomposition
import time
from einops import rearrange
from lovely_numpy import lo
from PIL import Image
from rich import print

# Internal imports.
from sgm.data import common

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
video_extensions = ['.mp4', '.avi', '.mov', '.flv', '.mkv', '.webm']


def load_model_bundle(device, config_path, model_path, support_ema,
                      num_steps=25, num_frames=14, max_scale=1.5, min_scale=1.0,
                      verbose=False):
    import torch
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config

    # Load inference config & diffusion model.
    test_config = OmegaConf.load(config_path)
    if 'cuda' in device:
        test_config.model.params.conditioner_config.params.emb_models[0].\
            params.open_clip_embedding_config.params.init_device = device
    test_config.model.params.ckpt_path = model_path

    # NOTE: This decides which keys to load and when, so it is important to get right!
    test_config.model.params.use_ema = bool(support_ema)
    test_config.model.params.ckpt_has_ema = bool(support_ema)

    # Here, we are setting the best known values so far to start off.
    test_config.model.params.sampler_config.params.num_steps = num_steps
    test_config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    test_config.model.params.sampler_config.params.guider_config.params.max_scale = max_scale
    test_config.model.params.sampler_config.params.guider_config.params.min_scale = min_scale
    test_config.model.params.sampler_config.params.device = device

    with torch.device(device):
        model = instantiate_from_config(test_config.model).to(device).eval()

    # Get train config to determine input & output modes.
    train_config_path = ''
    train_config = None
    try:
        train_config_path = model_path.replace('.ckpt', '.yaml')
        if not (os.path.exists(train_config_path)):
            train_config_path = str(pathlib.Path(model_path).parent)
            if os.path.basename(train_config_path) == 'checkpoints':
                train_config_path = str(pathlib.Path(train_config_path).parent)
            train_config_path = sorted(glob.glob(train_config_path + '/*config*/*.yaml'))[-1]
        train_config = OmegaConf.load(train_config_path)
    except:
        raise RuntimeError(f'Unable to load train config from within {train_config_path}.')

    if '/checkpoints' in model_path:
        # Most likely a custom training run.
        model_name = model_path.split('/checkpoints')[0].rsplit('_', 1)[-1]
    else:
        # Most likely a pretrained checkpoint provided by GCD authors.
        model_name = os.path.basename(model_path).split('.')[0]
        shortener = {'kubric': 'kb', 'pardom': 'pd',
                     'gradual': 'gr', 'direct': 'di',
                     'semantic': 'sem', 'max': 'm'}
        for (k, v) in shortener.items():
            model_name = model_name.replace(k, v)

    model_bundle = [model, train_config, test_config, device, model_name]
    model_bundle = expand_model_bundle(model_bundle, train_config, verbose=verbose)

    return model_bundle


def expand_model_bundle(model_bundle, train_config, verbose=True):

    azimuth_range = [0.0, 0.0]
    elevation_range = [0.0, 0.0]
    radius_range = [0.0, 0.0]
    trajectory = 'interpol_linear'
    move_time = 0
    camera_control = 'none'
    motion_bucket_range = [127, 127]

    if train_config is not None:
        if hasattr(train_config.data.params, 'azimuth_range'):
            azimuth_range = train_config.data.params.delta_azimuth_range
            if verbose:
                print(f'[cyan]Found azimuth range from train config: {azimuth_range}')
        if hasattr(train_config.data.params, 'elevation_range'):
            elevation_range = train_config.data.params.delta_elevation_range
            if verbose:
                print(f'[cyan]Found elevation range from train config: {elevation_range}')
        if hasattr(train_config.data.params, 'radius_range'):
            radius_range = train_config.data.params.delta_radius_range
            if verbose:
                print(f'[cyan]Found radius range from train config: {radius_range}')
        if hasattr(train_config.data.params, 'trajectory'):
            trajectory = train_config.data.params.trajectory
            if verbose:
                print(f'[cyan]Found camera trajectory from train config: {trajectory}')
        if hasattr(train_config.data.params, 'move_time'):
            move_time = train_config.data.params.move_time
            if verbose:
                print(f'[cyan]Found camera move time from train config: {move_time}')
        if hasattr(train_config.data.params, 'camera_control'):
            camera_control = train_config.data.params.camera_control
            if verbose:
                print(f'[cyan]Found camera control type from train config: {camera_control}')
        if hasattr(train_config.data.params, 'motion_bucket_range'):
            motion_bucket_range = train_config.data.params.motion_bucket_range
            if isinstance(motion_bucket_range, str):
                motion_bucket_range = list(map(int, motion_bucket_range.split(',')))
            else:
                motion_bucket_range = list(motion_bucket_range)
            if verbose:
                print(f'[cyan]Found motion bucket ID range from train config: '
                      f'{motion_bucket_range}')

    model_bundle += [azimuth_range, elevation_range, radius_range,
                     trajectory, move_time, camera_control, motion_bucket_range]

    return model_bundle


def warn_resolution_mismatch(train_config, frame_width, frame_height):
    model_trained_on = (train_config.data.params.frame_width,
                        train_config.data.params.frame_height)

    if model_trained_on[0] != frame_width or model_trained_on[1] != frame_height:
        print(f'[red]: Warning: '
              f'Desired input resolution mismatch: Model was trained on '
              f'{model_trained_on[0]} x {model_trained_on[1]} '
              f'but we are running inference at {frame_width} x {frame_height}!')


def prepare_model_inference_params(
        model, device, num_steps, num_frames, max_scale, min_scale, autocast, decoding_t):
    import torch

    # NOTE: adjusting the config probably has no effect anymore at this point.
    # test_config.model.params.sampler_config.params.num_steps = num_steps
    # test_config.model.params.sampler_config.params.guider_config.params.num_frames = \
    #     num_frames
    # test_config.model.params.sampler_config.params.guider_config.params.max_scale = \
    #     max_scale
    # test_config.model.params.sampler_config.params.guider_config.params.min_scale = \
    #     min_scale
    model.sampler.num_steps = num_steps
    model.sampler.guider.num_frames = num_frames
    model.sampler.guider.max_scale = max_scale
    model.sampler.guider.min_scale = min_scale

    model.en_and_decode_n_samples_a_time = decoding_t
    for embedder in model.conditioner.embedders:
        if hasattr(embedder, 'disable_encoder_autocast'):
            embedder.disable_encoder_autocast = not (bool(autocast))
        if hasattr(embedder, 'en_and_decode_n_samples_a_time'):
            embedder.en_and_decode_n_samples_a_time = decoding_t

    autocast_kwargs = {
        'enabled': bool(autocast),
        'device_type': device.split(':')[0],
        'dtype': torch.get_autocast_gpu_dtype(),
        'cache_enabled': torch.is_autocast_cache_enabled(),
    }

    return autocast_kwargs


def construct_batch(input_rgb, azimuth_deg, elevation_deg, radius_m,
                    input_frames, frame_rate, motion_bucket, cond_aug,
                    force_custom_mbid, model_bundle, device):
    '''
    This is mostly for arbitrary / custom videos (so not necessarily Kubric or ParallelDomain).
    '''
    import torch

    train_config = model_bundle[1]
    delta_azimuth_range = model_bundle[-7]
    delta_elevation_range = model_bundle[-6]
    delta_radius_range = model_bundle[-5]
    trajectory = model_bundle[-4]
    move_time = model_bundle[-3]
    camera_control = model_bundle[-2]
    motion_bucket_range = model_bundle[-1]

    (Tc, _, Hp, Wp) = input_rgb.shape
    input_rgb_torch = torch.tensor(input_rgb, device=device) * 2.0 - 1.0
    # (Tc, 3, Hp, Wp) tensor of float32 in [-1, 1].

    if input_frames < Tc:
        input_rgb_torch[input_frames:] = input_rgb_torch[input_frames - 1:input_frames]

    # This overlaps with infer.py.
    batch = dict()
    batch['motion_bucket_id'] = \
        torch.ones(Tc, dtype=torch.int32, device=device) * motion_bucket
    batch['fps_id'] = \
        torch.ones(Tc, dtype=torch.int32, device=device) * frame_rate
    batch['cond_aug'] = \
        torch.ones(Tc, dtype=torch.float32, device=device) * cond_aug
    batch['cond_frames_without_noise'] = input_rgb_torch
    batch['cond_frames'] = input_rgb_torch + \
        torch.randn_like(input_rgb_torch) * cond_aug
    batch['jpg'] = torch.zeros_like(input_rgb_torch)
    batch['image_only_indicator'] = \
        torch.zeros(1, Tc, dtype=torch.float32, device=device)
    batch['num_video_frames'] = Tc

    print(f'[cyan]Applying spherical camera trajectory: '
          f'azimuth={azimuth_deg}, elevation={elevation_deg}, radius={radius_m}.')

    # This overlaps with kubric_arbit.py.
    if camera_control == 'spherical':
        assert np.isfinite(azimuth_deg) and np.isfinite(elevation_deg) and np.isfinite(radius_m)
        spherical_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        spherical_end = np.array([azimuth_deg, elevation_deg, radius_m], dtype=np.float32)
        (spherical_src, spherical_dst) = common.construct_trajectory(
            spherical_start, spherical_end, trajectory, Tc, move_time)
        scaled_rel_angles = spherical_dst - spherical_src
        scaled_rel_angles[:, 0] *= np.pi / 180.0
        scaled_rel_angles[:, 1] *= np.pi / 180.0
        batch['scaled_relative_angles'] = \
            torch.tensor(scaled_rel_angles, dtype=torch.float32, device=device)

    # This overlaps with pardom_arbit.py.
    elif camera_control == 'relative_pose':
        scaled_rel_pose = torch.zeros((Tc, 3, 4), dtype=torch.float32)
        batch['scaled_relative_pose'] = scaled_rel_pose.type(torch.float32)

    # Overwrite motion value if model was trained with synchronized values.
    motion_range = motion_bucket_range[1] - motion_bucket_range[0]
    if camera_control != 'none' and not (force_custom_mbid) and motion_range > 0:
        my_motion = np.linalg.norm(spherical_end[0:2] - spherical_start[0:2], ord=2)
        max_motion = np.linalg.norm([max(*delta_azimuth_range),
                                    max(*delta_elevation_range)], ord=2)
        motion_amount = my_motion / max_motion
        motion_value = int(round(motion_bucket_range[0] +
                                 motion_range * motion_amount))
        batch['motion_bucket_id'] = torch.ones(Tc, dtype=torch.int32, device=device) * motion_value

    return batch


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


def is_image_folder(path):
    '''
    Determines whether the given directory mainly contains images, suggesting that they are frames
    of a video.
    '''
    files = [f for f in os.listdir(path)]
    num_image_files = sum([1 for f in files if os.path.isfile(os.path.join(path, f))
                           and os.path.splitext(f)[1].lower() in image_extensions])
    fraction = num_image_files > len(files) / 2
    return (fraction > 0.8)


def is_image_file(path):
    '''
    Determines whether the given file is an image file.
    '''
    return os.path.splitext(path)[1].lower() in image_extensions


def is_video_file(path):
    '''
    Determines whether the given file is a video file.
    '''
    return os.path.splitext(path)[1].lower() in video_extensions


def get_list_of_input_videos(paths):
    '''
    Returns a flattened list of examples, each of which can be a video file or a directory
    containing images which, in order, constitute frames of a video.
    :param paths: List of str.
    '''
    result = []

    for path in paths:
        if not (os.path.exists(path)):
            matches = glob.glob(path)
            result.extend(matches)

        elif os.path.isfile(path):
            # If this path is a video file, add it to the final list directly.
            if is_video_file(path):
                result.append(path)

            # If this path is instead a list of pointers to paths, recurse.
            if path.lower().endswith('.txt'):
                with open(path, 'r') as f:
                    lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if len(line) > 0]
                lines = [line for line in lines if not line.startswith('#')]
                result.extend(get_list_of_input_videos(lines))

        elif os.path.isdir(path):
            # If this path is a directory with frames, add it to the final list directly.
            if is_image_folder(path):
                result.append(path)

            # If this path is instead a directory of directories, recurse.
            else:
                dirs = list(sorted(os.listdir(path)))
                dirs = [os.path.join(path, f) for f in dirs]
                dirs = [f for f in dirs if os.path.isdir(f)]
                result.extend(get_list_of_input_videos(dirs))

    return result


def get_list_of_input_images_or_videos(paths):
    result = []

    for path in paths:
        if not (os.path.exists(path)):
            matches = glob.glob(path)
            result.extend(matches)

        elif os.path.isfile(path):
            # If this path is a video file, add it to the final list directly.
            if is_image_file(path) or is_video_file(path):
                result.append(path)

            # If this path is instead a list of pointers to paths, directly add them.
            if path.lower().endswith('.txt'):
                with open(path, 'r') as f:
                    lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if len(line) > 0]
                lines = [line for line in lines if not line.startswith('#')]
                result.extend(lines)

        elif os.path.isdir(path):
            # If this path is a directory with frames, add it to the final list directly.
            if is_image_folder(path):
                result.append(path)

            # If this path is instead a directory of directories, recurse.
            else:
                dirs = list(sorted(os.listdir(path)))
                dirs = [os.path.join(path, f) for f in dirs]
                dirs = [f for f in dirs if os.path.isdir(f)]
                result.extend(get_list_of_input_images_or_videos(dirs))

    return result


def load_video(src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial):
    '''
    :return rgb: (Tc, 3, Hp, Wp) array of float32 in [-1, 1].
    '''
    if os.path.isfile(src_path):
        rgb = common.load_video_mp4(
            src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial)
    elif os.path.isdir(src_path):
        if 'kubgen' in src_path:
            rgb = common.load_kubric_video_rgb_frames(
                src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial)
        else:
            rgb = common.load_video_all_frames(
                src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial)
    else:
        raise ValueError(f'Invalid path: {src_path}')
    return rgb


def load_image_or_video(
        src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial):
    '''
    :return rgb: (Tc, 3, Hp, Wp) array of float32 in [-1, 1].
    '''
    if is_image_file(src_path):
        rgb = common.load_rgb_image(
            src_path, center_crop, frame_width, frame_height, warn_spatial)
        rgb = rgb[None].repeat(len(clip_frames), axis=0)
    else:
        rgb = load_video(
            src_path, clip_frames, center_crop, frame_width, frame_height, warn_spatial)
    return rgb


def draw_text(image, position, anchor, caption, color, size_mult, darken_background=True):
    '''
    :param image (H, W, 3) array of float in [0, 1].
    :param position (2) tuple of int: (y, x) absolute coordinates of the anchor within the image.
    :param anchor (2) tuple of float: (y, x) relative coordinates of the anchor within the caption
        box. For example, (0, 0) means position corresponds to top-left corner of caption box;
        (1.0, 0.5) means position corresponds to bottom side of caption box.
    :param caption (str): Text to draw.
    :param color (3) tuple of float in [0, 1]: RGB values.
    :param size_mult (float): Multiplier for font size.
    :return image (H, W, 3) array of float in [0, 1].
    '''
    # Draw background and write text using OpenCV.
    label_width = int((8 + len(caption) * 9) * size_mult)
    label_height = int(21 * size_mult)
    (y, x) = (int(position[0]), int(position[1]))
    y -= int(anchor[0] * label_height)
    x -= int(anchor[1] * label_width)
    if image.dtype.kind == 'f':
        color = (float(color[0]), float(color[1]), float(color[2]))
    else:
        color = (int(color[0]), int(color[1]), int(color[2]))
    if darken_background:
        if image.dtype.kind == 'f':
            image[y:y + label_height, x:x + label_width] /= 3.0
        else:
            image[y:y + label_height, x:x + label_width] //= 3
    image = cv2.putText(image, caption, (x, y + label_height - int(7 * size_mult)), 2,
                        0.5 * size_mult, color, thickness=int(round(size_mult)))
    return image


def quick_pca(array, k=3, normalize=None):
    '''
    :param array (*, n): Array to perform PCA on.
    :param k (int) < n: Number of components to keep.
    :param normalize: (min, max) tuple of float: If not None, normalize output to this range.
    '''
    n = array.shape[-1]
    all_axes_except_last = tuple(range(len(array.shape) - 1))
    array_flat = array.reshape(-1, n)

    pca = sklearn.decomposition.PCA(n_components=k)
    pca.fit(array_flat)

    result_unnorm = pca.transform(array_flat).reshape(*array.shape[:-1], k)

    if normalize is not None:
        per_channel_min = result_unnorm.min(axis=all_axes_except_last, keepdims=True)
        per_channel_max = result_unnorm.max(axis=all_axes_except_last, keepdims=True)
        result = (result_unnorm - per_channel_min) / (per_channel_max - per_channel_min)
        result = result * (normalize[1] - normalize[0]) + normalize[0]

    else:
        result = result_unnorm

    result = result.astype(np.float32)
    return result


def write_video_and_frames(images, dst_dp=None, dst_fp=None, fps=10, save_images=True,
                           save_mp4=True, crop_multiple=None, quality=8):
    '''
    :param images (T, H, W, 3) array of float in [0, 1] or uint8 in [0, 255].
    '''
    if dst_dp is not None and dst_fp is None:
        dst_fp = dst_dp
    elif dst_fp is not None and dst_dp is None:
        dst_dp = os.path.splitext(dst_fp)[0] + '_frames'
    assert dst_dp is not None and dst_fp is not None

    if isinstance(images, list):
        images = np.stack(images)
    if images.dtype in [np.float16, np.float32, np.float64]:
        images = (images * 255.0).astype(np.uint8)

    if save_images:
        os.makedirs(dst_dp, exist_ok=True)
        print(f'[yellow]Saving frames as images to: {dst_dp}')
        for i, image in enumerate(images):
            plt.imsave(os.path.join(dst_dp, f'{i:04d}.png'), image)

    if save_mp4:
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        os.makedirs(str(pathlib.Path(dst_fp).parent), exist_ok=True)

        # After saving images, we can adjust dimensions to respect video codec blocks.
        if crop_multiple is not None and crop_multiple > 1:
            (H, W) = images.shape[1:3]
            H_crop = crop_multiple * (H // crop_multiple)
            W_crop = crop_multiple * (W // crop_multiple)
            images = images[:, 0:H_crop, 0:W_crop]

        images = list(images)
        video_frames = images

        def final_step():
            # Force affinity to cover all CPU cores.
            cpu_count = os.cpu_count()
            affinity = set(range(0, cpu_count))
            os.sched_setaffinity(0, affinity)
            print(f'[yellow]Saving video to: {dst_fp}.mp4')

            max_attempts = 10
            for i in range(max_attempts):
                try:
                    # NOTE: This mysteriously sometimes fails with:
                    # FileNotFoundError: [Errno 2] No such file or directory.
                    imageio.mimwrite(dst_fp + '.mp4', video_frames, format='ffmpeg', fps=float(fps),
                                     macro_block_size=crop_multiple, quality=quality)
                    break  # Success.

                except Exception as e:
                    print(f'[red]Error saving video: {e}')
                    if i <= max_attempts - 2:
                        print(f'[red]Retrying (attempt {i + 1})...')
                    time.sleep(1.0)

        final_step()


def masked_ssim(im1, im2, mask, win_size=7, K1=0.01, K2=0.03, sigma=1.5, channel_axis=0):
    '''
    This is adapted from scikit-learn version 0.22.0 skimage.metrics.structural_similarity,
    but we allow for arbitrary non-rectangular regions to be considered only.
    NOTE: We assume these parameters:
    data_range = 1.0, full = False, gaussian_weights = False, gradient = False.
    :param im1: (C?, H, W, C?) array of float in [0, 1].
    :param im2: (C?, H, W, C?) array of float in [0, 1].
    :param mask: (H, W) array of bool.
    :return (mssim_all, mssim_mask): float x2.
    '''

    from scipy.ndimage import binary_erosion, uniform_filter
    from skimage._shared import utils
    from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
    from skimage.util.arraycrop import crop

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    mask = mask.astype(bool)
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    if channel_axis is not None:
        # loop over channels
        nch = im1.shape[channel_axis]

        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)

        mssims = []
        for ch in range(nch):
            ch_result = masked_ssim(
                im1[_at(ch)], im2[_at(ch)], mask, win_size=win_size, K1=K1, K2=K2, sigma=sigma,
                channel_axis=None)
            mssims.append(ch_result)

        mssims = np.mean(mssims, axis=0)
        return mssims

    use_sample_covariance = True
    ndim = im1.ndim
    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = 1.0
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # OLD:
    # compute (weighted) mean of ssim. Use float64 for accuracy.
    S_crop = crop(S, pad)
    mssim_all = np.mean(S_crop, dtype=np.float64)

    # NEW:
    mask_erode = binary_erosion(mask, iterations=pad)
    mask_crop = crop(mask_erode, pad)
    mssim_mask = np.mean(S_crop[mask_crop], dtype=np.float64)

    mssims = np.array([mssim_all, mssim_mask])
    return mssims


def save_video(dst_fp, frames, fps, quality):
    import torch
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    if frames.dtype.kind == 'f':
        frames = (frames * 255.0).astype(np.uint8)
    imageio.mimwrite(dst_fp, frames, format='ffmpeg', fps=float(fps),
                     macro_block_size=8, quality=int(quality))


class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).

            for (cone, clr, legend) in [(input_cone, 'green', 'Input view'),
                                        (output_cone, 'blue', 'Target view')]:

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 0)))
                    # text=(legend if i == 0 else None),
                    # textposition='bottom center'))
                    # hoverinfo='text',
                    # hovertext='hovertext'))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T
