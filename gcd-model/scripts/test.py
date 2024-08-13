'''
Created by Basile Van Hoorick for GCD, 2024.
Evaluate model on one or multiple scenes from Kubric-4D or ParallelDomain-4D.
'''

import os  # noqa
import sys  # noqa
sys.path.insert(0, os.getcwd())  # noqa

# Library imports.
import argparse
import copy
import cv2
import glob
import joblib
import lovely_tensors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pathlib
import random
import skimage
import skimage.metrics
import sys
import time
import tqdm
import tqdm.rich
import traceback
import warnings
from einops import rearrange
from lovely_numpy import lo
from rich import print
from tqdm import TqdmExperimentalWarning

# Internal imports.
from scripts import eval_utils

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)


def test_args():

    parser = argparse.ArgumentParser()

    # Resource options.
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--debug', type=int, default=0)

    # General data options.
    parser.add_argument('--input', type=str, nargs='+',
                        default=[r'../eval/list/cool_videos.txt'],
                        help='One or more paths to video files, and/or directories with images, '
                        'and/or root of evaluation sets, and/or text files with list of examples.')
    parser.add_argument('--output', type=str,
                        default=r'../eval/output/dbg1')

    # General model options.
    parser.add_argument('--config_path', type=str,
                        default=r'configs/infer_kubric.yaml')
    parser.add_argument('--model_path', type=str, nargs='+',
                        default=[r'../logs/*_kb_v1/checkpoints/last.ckpt'],
                        help='One or more paths to trained model weights.')
    parser.add_argument('--use_ema', type=int, default=0)
    parser.add_argument('--autocast', type=int, default=1)

    # Model inference options.
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=14)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--guider_max_scale', type=float, default=1.5)
    parser.add_argument('--guider_min_scale', type=float, default=1.0)
    parser.add_argument('--motion_id', type=int, default=127)
    # ^ NOTE: If motion_bucket_id is synchronized with camera angles during training, this code
    # will take care of automatically setting it (thus overriding the provided value).
    parser.add_argument('--cond_aug', type=float, default=0.02)
    parser.add_argument('--decoding_t', type=int, default=14)

    # Camera control & frame bounds options.
    parser.add_argument('--control_json', type=str, default='')
    parser.add_argument('--control_idx', type=int, default=0)
    parser.add_argument('--azimuth_start', type=float, default=20.0)
    parser.add_argument('--elevation_start', type=float, default=10.0)
    parser.add_argument('--radius_start', type=float, default=15.0)
    parser.add_argument('--delta_azimuth', type=float, default=30.0)
    parser.add_argument('--delta_elevation', type=float, default=15.0)
    parser.add_argument('--delta_radius', type=float, default=0.0)
    parser.add_argument('--frame_start', type=int, default=-1)
    parser.add_argument('--frame_stride', type=int, default=-1)
    parser.add_argument('--frame_rate', type=int, default=-1)

    # Data processing options.
    parser.add_argument('--frame_width', type=int, default=384)
    parser.add_argument('--frame_height', type=int, default=256)
    parser.add_argument('--center_crop', type=int, default=1)
    parser.add_argument('--save_images', type=int, default=1)
    parser.add_argument('--save_mp4', type=int, default=1)
    # ^ NOTE: Galleries always have MP4 regardless of the save_mp4 setting.
    parser.add_argument('--save_input', type=int, default=1)
    parser.add_argument('--save_gt', type=int, default=1)
    parser.add_argument('--save_error', type=int, default=1)
    parser.add_argument('--save_uncertainty', type=int, default=1)
    parser.add_argument('--reproject_rgbd', type=int, default=1)
    # ^ NOTE: This is necessary for occluded or visible metrics.

    # Metrics options.
    parser.add_argument('--calculate_metrics', type=int, default=1)

    args = parser.parse_args()

    args.gpus = [int(x.strip()) for x in args.gpus.split(',')]

    return args


def load_input_gt(args, worker_idx, example, train_config, control_info, device):
    '''
    :return input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].
    :return gt_rgb: (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].
    :return controls (dict).
    :return batch (dict).
    '''
    import torch
    from sgm.util import instantiate_from_config

    example_base = os.path.basename(example)
    example_base2 = os.path.basename(os.path.dirname(example))
    example_base3 = os.path.basename(os.path.dirname(os.path.dirname(example)))

    control_angles = (args.azimuth_start > -1000)
    if control_angles:
        controls = np.array([args.frame_start, args.frame_stride, args.frame_rate,
                            args.azimuth_start, args.azimuth_start + args.delta_azimuth,
                            args.elevation_start, args.elevation_start + args.delta_elevation,
                            args.radius_start, args.radius_start + args.delta_radius],
                            dtype=np.float32)
    else:
        controls = np.array([args.frame_start, args.frame_stride, args.frame_rate,
                            -2222, -2222, -2222, -2222, -2222, -2222],
                            dtype=np.float32)
    # NOTE: ^ All values are in degrees.

    my_dmod = None
    my_dset = None
    batch = None
    gt_rgb = None
    reproject_rgb = None

    if 'kubric' in train_config.data.target:

        train_config.data.target = 'sgm.data.kubric_arbit.KubricSynthViewModule'
        train_config.data.params.frame_width = args.frame_width
        train_config.data.params.frame_height = args.frame_height
        train_config.data.params.cond_aug = args.cond_aug

        my_dmod = instantiate_from_config(train_config.data)
        my_dset = my_dmod.val_dataset
        my_dset.model_frames = args.num_frames
        my_dset.input_frames = args.num_frames
        my_dset.output_frames = args.num_frames
        my_dset.data_gpu = int(device.split(':')[1])
        my_dset.max_retries = 10

        if 'scn' in example_base:
            scene_idx = int(example_base[3:])
        else:
            raise ValueError(f'Unknown Kubric scene: {example_base}')

        if control_info is not None:
            print(f'[gray]{worker_idx}: Applying Kubric-4D control entry for {example_base}...')

            sample_name = f'sample_{args.control_idx:02d}'
            control_entry = control_info[example_base][sample_name]

            # NOTE: Angles will always be overwritten by controls,
            # but clip bounds can still be overridden by command line.
            controls[3:9] = [control_entry['spherical_start'][0], control_entry['spherical_end'][0],
                             control_entry['spherical_start'][1], control_entry['spherical_end'][1],
                             control_entry['spherical_start'][2], control_entry['spherical_end'][2]]

            if args.frame_start < 0:
                controls[0] = control_entry['frame_start']
            if args.frame_stride < 0:
                controls[1] = control_entry['frame_skip']
            if args.frame_rate < 0:
                controls[2] = int(round(24 / controls[1]))  # This corresponds to fps_id.

        else:
            assert args.frame_start >= 0 and args.frame_stride >= 0 and args.frame_rate >= 0, \
                f'{args.frame_start} {args.frame_stride} {args.frame_rate}'

        # KubricSynthViewDataset expects: [scene_idx, frame_skip, frame_start, reverse,
        # azimuth_start, azimuth_end, elevation_start, elevation_end, radius_start, radius_end].
        my_dset.set_next_example(
            scene_idx, int(controls[1]), int(controls[0]), False, *controls[3:9])
        my_dset.reproject_rgbd = bool(args.reproject_rgbd)

        batch = my_dset[0]
        batch = {k: torch.tensor(v, device=device) for (k, v) in batch.items()}
        batch['num_video_frames'] = args.num_frames

        # Only if needed, update the control values to reflect the randomly sampled metadata.
        if not (control_angles):
            controls[3] = 0.0
            controls[4] = batch['scaled_relative_angles'][-1][0] * 180.0 / np.pi
            controls[5] = 0.0
            controls[6] = batch['scaled_relative_angles'][-1][1] * 180.0 / np.pi
            controls[7] = 0.0
            controls[8] = batch['scaled_relative_angles'][-1][2]

        input_rgb = batch['cond_frames_without_noise'].detach().cpu().numpy()
        gt_rgb = batch['jpg'].detach().cpu().numpy()
        # 2x (Tco, 3, Hp, Wp) array of float32 in [-1, 1].

        if args.reproject_rgbd:
            if 'reproject' in batch:
                reproject_rgb = batch['reproject'].detach().cpu().numpy()
                # (Tco, 3, Hp, Wp) array of float32 in [-1, 1].
            else:
                print(f'[red]{worker_idx}: Warning: reproject not found in batch!')

    elif 'pardom' in train_config.data.target:

        train_config.data.target = 'sgm.data.pardom_arbit.ParallelDomainSynthViewModule'
        train_config.data.params.frame_width = args.frame_width
        train_config.data.params.frame_height = args.frame_height
        train_config.data.params.cond_aug = args.cond_aug

        my_dmod = instantiate_from_config(train_config.data)
        my_dset = my_dmod.val_dataset
        my_dset.model_frames = args.num_frames
        my_dset.input_frames = args.num_frames
        my_dset.output_frames = args.num_frames
        my_dset.data_gpu = int(device.split(':')[1])
        my_dset.max_retries = 10

        # No camera controls are currently used for ParallelDomain.
        controls[3:9] = 0.0

        if 'scene' in example_base:
            scene_idx = int(example_base[6:])
        else:
            raise ValueError(f'Unknown ParallelDomain scene: {example_base}')

        if control_info is not None:
            print(f'[gray]{worker_idx}: Applying ParallelDomain-4D control entry for {example_base}...')

            sample_name = f'sample_{args.control_idx:02d}'
            control_entry = control_info[example_base][sample_name]

            if args.frame_start < 0:
                controls[0] = control_entry['frame_start']
            if args.frame_stride < 0:
                controls[1] = control_entry['frame_skip']
            if args.frame_rate < 0:
                controls[2] = int(round(10 / controls[1]))  # This corresponds to fps_id.

        else:
            assert args.frame_start >= 0 and args.frame_stride >= 0 and args.frame_rate >= 0, \
                f'{args.frame_start} {args.frame_stride} {args.frame_rate}'

        # ParallelDomainSynthViewDataset expects: [scene_idx, scene_dn, frame_skip, frame_start,
        # reverse].
        my_dset.set_next_example(
            scene_idx, example_base, int(controls[1]), int(controls[0]), False)
        my_dset.reproject_rgbd = bool(args.reproject_rgbd)

        batch = my_dset[0]
        batch = {k: torch.tensor(v, device=device) for (k, v) in batch.items()}
        batch['num_video_frames'] = args.num_frames

        # NOTE: We update the control values to reflect the randomly sampled metadata. This works
        # quite differently from Kubric where the controls actually define the trajectory, but this
        # is not supported here; rather, these values are purely informative.
        controls[4] = batch['scaled_relative_angles'][-1][0] * 180.0 / np.pi
        controls[6] = batch['scaled_relative_angles'][-1][1]
        controls[8] = batch['scaled_relative_angles'][-1][2]

        input_rgb = batch['cond_frames_without_noise'].detach().cpu().numpy()
        gt_rgb = batch['jpg'].detach().cpu().numpy()
        # 2x (Tco, 3, Hp, Wp) array of float32 in [-1, 1].

        if args.reproject_rgbd:
            if 'reproject' in batch:
                reproject_rgb = batch['reproject'].detach().cpu().numpy()
                # (Tco, 3, Hp, Wp) array of float32 in [-1, 1].
            else:
                print(f'[red]{worker_idx}: Warning: reproject not found in batch!')

    else:
        raise ValueError(f'Config exists but refers to unknown training dataset: '
                         f'{train_config.data.target}')

    # Finally, apply data range correction such that we are consistent with sample_video().
    input_rgb = (input_rgb + 1.0) / 2.0
    # (Tci, 3, Hp, Wp) array of float32 in [0, 1].
    gt_rgb = (gt_rgb + 1.0) / 2.0
    # (Tco, 3, Hp, Wp) array of float32 in [0, 1].
    if reproject_rgb is not None:
        reproject_rgb = (reproject_rgb + 1.0) / 2.0
        # (Tco, 3, Hp, Wp) array of float32 in [0, 1].

    (_, _, Hp, Wp) = input_rgb.shape
    assert Hp % 64 == 0 and Wp % 64 == 0, \
        f'Input resolution must be a multiple of 64, but got {Hp} x {Wp}'

    return (input_rgb, gt_rgb, reproject_rgb, controls, batch)


def run_inference(args, device, model, batch):
    import torch

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, device, args.num_steps, args.num_frames,
        args.guider_max_scale, args.guider_min_scale, args.autocast, args.decoding_t)

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            pred_samples = []

            for sample_idx in range(args.num_samples):
                # Perform denoising loop.
                # NOTE: use_ema is False because we already entered the EMA scope before
                # (i.e. when calling process_example which calls run_inference).
                video_dict = model.sample_video(
                    batch, enter_ema=False, limit_batch=False)

                output_dict = dict()
                output_dict['cond_rgb'] = video_dict['cond_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_rgb'] = video_dict['sampled_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_latent'] = video_dict['sampled_z'].detach().cpu().numpy()
                # (Tcm, 4, Hl, Wl) = (14, 4, 32, 48) array of float32.

                pred_samples.append(output_dict)

    return pred_samples


def calculate_metrics(args, gt_rgb, reproject_rgb, pred_samples):
    '''
    :param input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param gt_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param reproject_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param pred_samples: list of dicts with keys:
        cond_rgb, sampled_rgb, sampled_latent.
    '''
    # NOTE: This subroutine is a bit rudimentary, because it does not include baseline metrics;
    # see more advanced scripts for that.
    S = len(pred_samples)

    if S >= 1:
        pred_samples_rgb = np.stack([x['sampled_rgb'] for x in pred_samples], axis=0)
        # (S, Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    else:
        pred_samples_rgb = []

    if reproject_rgb is not None:
        occluded_mask = (np.sum(np.abs(reproject_rgb), axis=1) <= 1e-7).astype(np.uint8)
        visible_mask = 1 - occluded_mask
        # 2x (Tcm, Hp, Wp) array of uint8 in [0, 1].

        visible_mask_bc = np.tile(visible_mask[:, None].astype(bool), (1, 3, 1, 1))
        occluded_mask_bc = np.tile(occluded_mask[:, None].astype(bool), (1, 3, 1, 1))
        # 2x (Tcm, 3, Hp, Wp) array of bool.

    frame_psnr = []
    frame_psnr_vis = []
    frame_psnr_occ = []
    frame_ssim = []
    frame_ssim_vis = []
    frame_ssim_occ = []

    for output_rgb in pred_samples_rgb:
        # output_rgb = (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
        (Tcm, _, Hp, Wp) = output_rgb.shape

        cur_frame_psnr = []
        cur_frame_psnr_vis = []
        cur_frame_psnr_occ = []
        cur_frame_ssim = []
        cur_frame_ssim_vis = []
        cur_frame_ssim_occ = []

        for t in range(Tcm):

            cur_psnr = skimage.metrics.peak_signal_noise_ratio(
                output_rgb[t], gt_rgb[t], data_range=1.0)
            cur_ssim = skimage.metrics.structural_similarity(
                output_rgb[t], gt_rgb[t], data_range=1.0, channel_axis=0)

            cur_frame_psnr.append(cur_psnr)
            cur_frame_ssim.append(cur_ssim)

            if reproject_rgb is not None:
                cur_vis_mask = visible_mask_bc[t]  # (3, Hp, Wp) array of bool.
                cur_occ_mask = occluded_mask_bc[t]  # (3, Hp, Wp) array of bool.
                cur_output_vis = output_rgb[t][cur_vis_mask]
                cur_gt_vis = gt_rgb[t][cur_vis_mask]
                cur_output_occ = output_rgb[t][cur_occ_mask]
                cur_gt_occ = gt_rgb[t][cur_occ_mask]

                if cur_vis_mask.any():
                    cur_psnr_vis = skimage.metrics.peak_signal_noise_ratio(
                        cur_output_vis, cur_gt_vis, data_range=1.0)
                    cur_ssim_vis = eval_utils.masked_ssim(
                        output_rgb[t], gt_rgb[t], cur_vis_mask[0])[1]
                else:
                    cur_psnr_vis = np.nan
                    cur_ssim_vis = np.nan

                if cur_occ_mask.any():
                    cur_psnr_occ = skimage.metrics.peak_signal_noise_ratio(
                        cur_output_occ, cur_gt_occ, data_range=1.0)
                    cur_ssim_occ = eval_utils.masked_ssim(
                        output_rgb[t], gt_rgb[t], cur_occ_mask[0])[1]
                else:
                    cur_psnr_occ = np.nan
                    cur_ssim_occ = np.nan

                cur_frame_psnr_vis.append(cur_psnr_vis)
                cur_frame_ssim_vis.append(cur_ssim_vis)
                cur_frame_psnr_occ.append(cur_psnr_occ)
                cur_frame_ssim_occ.append(cur_ssim_occ)

        frame_psnr.append(cur_frame_psnr)
        frame_ssim.append(cur_frame_ssim)
        frame_psnr_vis.append(cur_frame_psnr_vis)
        frame_ssim_vis.append(cur_frame_ssim_vis)
        frame_psnr_occ.append(cur_frame_psnr_occ)
        frame_ssim_occ.append(cur_frame_ssim_occ)

    frame_psnr = np.array(frame_psnr)  # (S, Tcm) array of float.
    frame_ssim = np.array(frame_ssim)  # (S, Tcm) array of float.
    frame_psnr_vis = np.array(frame_psnr_vis)  # (S, Tcm) array of float.
    frame_ssim_vis = np.array(frame_ssim_vis)  # (S, Tcm) array of float.
    frame_psnr_occ = np.array(frame_psnr_occ)  # (S, Tcm) array of float.
    frame_ssim_occ = np.array(frame_ssim_occ)  # (S, Tcm) array of float.

    mean_psnr = np.nanmean(frame_psnr, axis=1)  # (S) array of float.
    mean_ssim = np.nanmean(frame_ssim, axis=1)  # (S) array of float.
    mean_psnr_vis = np.nanmean(frame_psnr_vis, axis=1)  # (S) array of float.
    mean_ssim_vis = np.nanmean(frame_ssim_vis, axis=1)  # (S) array of float.
    mean_psnr_occ = np.nanmean(frame_psnr_occ, axis=1)  # (S) array of float.
    mean_ssim_occ = np.nanmean(frame_ssim_occ, axis=1)  # (S) array of float.

    uncertainty = np.nanmean(np.std(pred_samples_rgb, axis=0), axis=1)
    # (Tcm, Hp, Wp) array of float32 in [0, 1].
    frame_diversity = np.nanmean(uncertainty, axis=(1, 2))
    # (Tcm) array of float32 in [0, 1].
    mean_diversity = np.nanmean(frame_diversity)
    # single float.

    if reproject_rgb is not None:
        # NOTE: To ensure correct statistics, we apply array masking instead of multiplication here.
        pred_samples_vis = [np.stack([x[t][visible_mask_bc[t]] for x in pred_samples_rgb], axis=0)
                            for t in range(Tcm)]
        pred_samples_occ = [np.stack([x[t][occluded_mask_bc[t]] for x in pred_samples_rgb], axis=0)
                            for t in range(Tcm)]
        # 2x List-T of (S, N) of float32 in [0, 1].
        frame_diversity_vis = np.array([np.nanmean(np.std(x, axis=0)) for x in pred_samples_vis])
        frame_diversity_occ = np.array([np.nanmean(np.std(x, axis=0)) for x in pred_samples_occ])
        # 2x (Tcm) array of float32 in [0, 1].
        mean_diversity_vis = np.nanmean(frame_diversity_vis)
        mean_diversity_occ = np.nanmean(frame_diversity_occ)
        # 2x single float.

    metrics_dict = dict()
    metrics_dict['frame_psnr'] = frame_psnr
    metrics_dict['frame_ssim'] = frame_ssim
    metrics_dict['frame_diversity'] = frame_diversity
    metrics_dict['mean_psnr'] = mean_psnr
    metrics_dict['mean_ssim'] = mean_ssim
    metrics_dict['mean_diversity'] = mean_diversity

    if reproject_rgb is not None:
        metrics_dict['frame_psnr_vis'] = frame_psnr_vis
        metrics_dict['frame_ssim_vis'] = frame_ssim_vis
        metrics_dict['frame_psnr_occ'] = frame_psnr_occ
        metrics_dict['frame_ssim_occ'] = frame_ssim_occ
        metrics_dict['frame_diversity_vis'] = frame_diversity_vis
        metrics_dict['frame_diversity_occ'] = frame_diversity_occ
        metrics_dict['mean_psnr_vis'] = mean_psnr_vis
        metrics_dict['mean_ssim_vis'] = mean_ssim_vis
        metrics_dict['mean_psnr_occ'] = mean_psnr_occ
        metrics_dict['mean_ssim_occ'] = mean_ssim_occ
        metrics_dict['mean_diversity_vis'] = mean_diversity_vis
        metrics_dict['mean_diversity_occ'] = mean_diversity_occ

    return (metrics_dict, uncertainty)


def get_controls_friendly(controls):
    frame_start = int(controls[0])
    frame_stride = int(controls[1])
    frame_rate = int(controls[2])
    delta_azimuth = controls[4] - controls[3]
    delta_elevation = controls[6] - controls[5]
    delta_radius = controls[8] - controls[7]
    # NOTE: ^ All values are in degrees.

    if delta_azimuth != 0.0 or delta_elevation != 0.0 or delta_radius != 0.0:
        nonzero = True
        title = f'A {delta_azimuth:.1f} E {delta_elevation:.1f} R {delta_radius:.1f}'
        filename = (f'fs{frame_start}_fr{frame_rate}_az{delta_azimuth:.1f}'
                    f'_el{delta_elevation:.1f}_rd{delta_radius:.1f}')

    else:
        nonzero = False
        title = f'FPS {frame_rate}'
        filename = f'fs{frame_start}_fr{frame_rate}'

    return (nonzero, title, filename)


def create_visualizations(
        args, input_rgb, gt_rgb, reproject_rgb, controls_friendly, pred_samples,
        metrics_dict, uncertainty, model_name):
    '''
    :param input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param gt_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param reproject_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param pred_samples: List of dict.
    :param uncertainty: (Tcm, Hp, Wp) array of float32 in [0, 1].
    '''
    (Tcm, _, Hp, Wp) = input_rgb.shape
    S = len(pred_samples)

    if controls_friendly[0]:
        target_title = f'Target ({controls_friendly[1]})'
    else:
        target_title = f'Target'

    input_rgb = rearrange(input_rgb, 't c h w -> t h w c')
    # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].
    gt_rgb = rearrange(gt_rgb, 't c h w -> t h w c')
    # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].
    if reproject_rgb is not None:
        reproject_rgb = rearrange(reproject_rgb, 't c h w -> t h w c')
        # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

    pred_samples_rgb = []
    error_rgb = None
    uncertainty_rgb = None

    if S >= 1:
        frame_psnr = metrics_dict['frame_psnr']
        frame_ssim = metrics_dict['frame_ssim']
        frame_diversity = metrics_dict['frame_diversity']

        if reproject_rgb is not None:
            frame_psnr_vis = metrics_dict['frame_psnr_vis']
            frame_psnr_occ = metrics_dict['frame_psnr_occ']
            frame_ssim_vis = metrics_dict['frame_ssim_vis']
            frame_ssim_occ = metrics_dict['frame_ssim_occ']

        pred_samples_rgb = [
            rearrange(x['sampled_rgb'], 't c h w -> t h w c') for x in pred_samples]
        # (S, Tcm, Hp, Wp, 3) array of float32 in [0, 1].
        pred_samples_latent = [
            rearrange(x['sampled_latent'], 't c h w -> t h w c') for x in pred_samples]
        # (S, Tcm, Hl, Wl, 4) array of float32.

        used_error = np.mean(np.abs(gt_rgb - pred_samples_rgb[0]), axis=-1)
        # (Tcm, Hp, Wp) array of float32 in [0, 1].
        error_rgb = plt.cm.magma(used_error)[..., 0:3]
        # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

        if uncertainty is not None:
            used_uncertainty = np.clip(uncertainty * 3.0, 0.0, 1.0)
            # (Tcm, Hp, Wp) array of float32 in [0, 1].
            uncertainty_rgb = plt.cm.magma(used_uncertainty)[..., 0:3]
            # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

        # NOTE: The PCA visualization is computed on all samples together to allow for comparison.
        pred_samples_latent_pca = eval_utils.quick_pca(
            np.stack(pred_samples_latent, axis=0), k=3, normalize=[0.0, 1.0])
        # (S, Tcm, Hl, Wl, 3) array of float32 in [0, 1].

        (_, Hl, Wl, _) = pred_samples_latent[0].shape
        F = Hp // Hl
        pred_samples_latent_vis = np.repeat(pred_samples_latent_pca, F, axis=2)
        pred_samples_latent_vis = np.repeat(pred_samples_latent_vis, F, axis=3)
        assert pred_samples_latent_vis.shape == (S, Tcm, Hp, Wp, 3)

    # NOTE: I disabled some of these for GCD, 2024 code publication since it might be a bit excessive for
    # most use cases, but feel free to re-enable them and/or implement your own stuff.

    rich1_frames = []
    rich2_frames = []
    rich3_frames = []
    rich4_frames = []
    rich5_frames = []
    rich6_frames = []
    rich7_frames = []
    font_size = 1.0

    for t in range(Tcm):
        # Rich 1: Input, Target || Output 1, Output 2 || Output 3, Output 4.
        if S <= 2:
            canvas1 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
        else:
            canvas1 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)

        eval_utils.draw_text(canvas1, (20, 5), (0.5, 0.0),
                             f'Input (Frame {t})', (1, 1, 1), font_size)
        eval_utils.draw_text(canvas1, (Hp + 60, 5), (0.5, 0.0),
                             target_title, (1, 1, 1), font_size)

        canvas1[40:Hp + 40, 0:Wp] = input_rgb[t]
        canvas1[Hp + 80:Hp * 2 + 80, 0:Wp] = gt_rgb[t].copy()

        if S >= 1:
            eval_utils.draw_text(
                canvas1, (20, Wp + 5), (0.5, 0.0),
                f'Output 1 (PSNR {frame_psnr[0, t]:.2f}, SSIM {frame_ssim[0, t]:.3f})',
                (1, 1, 1), font_size)
            canvas1[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t].copy()

        if S >= 2:
            eval_utils.draw_text(
                canvas1, (Hp + 60, Wp + 5), (0.5, 0.0),
                f'Output 2 (PSNR {frame_psnr[1, t]:.2f}, SSIM {frame_ssim[1, t]:.3f})',
                (1, 1, 1), font_size)
            canvas1[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] = pred_samples_rgb[1][t].copy()

        if S >= 3:
            eval_utils.draw_text(
                canvas1, (20, Wp * 2 + 5), (0.5, 0.0),
                f'Output 3 (PSNR {frame_psnr[2, t]:.2f}, SSIM {frame_ssim[2, t]:.3f})',
                (1, 1, 1), font_size)
            canvas1[40:Hp + 40, Wp * 2:Wp * 3] = pred_samples_rgb[2][t].copy()

        if S >= 4:
            eval_utils.draw_text(
                canvas1, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
                f'Output 4 (PSNR {frame_psnr[3, t]:.2f}, SSIM {frame_ssim[3, t]:.3f})',
                (1, 1, 1), font_size)
            canvas1[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = pred_samples_rgb[3][t].copy()

        rich1_frames.append(canvas1)

        # Rich 2: Input || Output 1.
        if S >= 1:
            canvas2 = canvas1[0:Hp + 40, 0:Wp * 2].copy()
            canvas2[0:40, Wp:Wp * 2] = 0.0
            eval_utils.draw_text(canvas2, (20, Wp + 5), (0.5, 0.0),
                                 f'Output ({model_name})', (1, 1, 1), font_size)

            rich2_frames.append(canvas2)

        # Rich 3: Input, Target || Output 1, Output 2 || Error 1, Error 2.
        if S >= 1:
            canvas3 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)
            canvas3[:, 0:Wp * 2] = canvas1[:, 0:Wp * 2].copy()

            if S >= 1:
                eval_utils.draw_text(canvas3, (20, Wp * 2 + 5), (0.5, 0.0),
                                     f'Error 1', (1, 1, 1), font_size)
                canvas3[40:Hp + 40, Wp * 2:Wp * 3] = \
                    np.abs(gt_rgb[t] - pred_samples_rgb[0][t])

            if S >= 2:
                eval_utils.draw_text(canvas3, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
                                     f'Error 2', (1, 1, 1), font_size)
                canvas3[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = \
                    np.abs(gt_rgb[t] - pred_samples_rgb[1][t])

            rich3_frames.append(canvas3)

        # Rich 4: Input, Target || Output 1, Output 2 || Latent 1, Latent 2.
        # if S >= 1:
        #     canvas4 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)
        #     canvas4[:, 0:Wp * 2] = canvas1[:, 0:Wp * 2].copy()

        #     eval_utils.draw_text(canvas4, (20, Wp * 2 + 5), (0.5, 0.0),
        #                          f'Latent 1', (1, 1, 1), font_size)
        #     canvas4[40:Hp + 40, Wp * 2:Wp * 3] = pred_samples_latent_vis[0][t].copy()

        #     if S >= 2:
        #         eval_utils.draw_text(canvas4, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
        #                              f'Latent 2', (1, 1, 1), font_size)
        #         canvas4[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = pred_samples_latent_vis[1][t].copy()

        #     rich4_frames.append(canvas4)

        # Rich 5: Input, Target ||  Delta, Uncert.
        if S >= 2 and uncertainty_rgb is not None:
            delta_rgb = np.abs(pred_samples_rgb[0][t] - pred_samples_rgb[1][t]) * 2.0
            canvas5 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
            canvas5[:, 0:Wp] = canvas1[:, 0:Wp].copy()

            eval_utils.draw_text(canvas5, (20, Wp + 5), (0.5, 0.0),
                                 f'Delta (Div {frame_diversity[t]:.3f})', (1, 1, 1), font_size)
            canvas5[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t] * 0.3  # Darken output.
            canvas5[40:Hp + 40, Wp:Wp * 2] += delta_rgb * 0.8

            eval_utils.draw_text(canvas5, (Hp + 60, Wp + 5), (0.5, 0.0),
                                 f'Uncertainty (Div {frame_diversity[t]:.3f})', (1, 1, 1), font_size)
            canvas5[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] = pred_samples_rgb[1][t] * 0.3  # Darken output.
            canvas5[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] += uncertainty_rgb[t] * 0.8

            rich5_frames.append(canvas5)

        # Rich 6: Input, Target || Output 1, Reproj.
        if S >= 1 and reproject_rgb is not None:
            canvas6 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
            canvas6[:, 0:Wp] = canvas1[:, 0:Wp].copy()

            eval_utils.draw_text(
                canvas6, (20, Wp + 5), (0.5, 0.0),
                f'Output 1 (PSNR Occ {frame_psnr_occ[0, t]:.2f}, SSIM Occ {frame_ssim_occ[0, t]:.2f})',
                (1, 1, 1), font_size)
            canvas6[40:Hp + 40, Wp:Wp * 2] = canvas1[40:Hp + 40, Wp:Wp * 2].copy()

            eval_utils.draw_text(
                canvas6, (Hp + 60, Wp + 5), (0.5, 0.0),
                f'Reproj (PSNR Vis {frame_psnr_vis[0, t]:.2f}, SSIM Vis {frame_ssim_vis[0, t]:.2f})',
                (1, 1, 1), font_size)
            canvas6[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] = reproject_rgb[t].copy()

            rich6_frames.append(canvas6)

        # Rich 7: Input, Target || Reproj + Error, Reproj + Uncert.
        # if args.reproject_rgbd and S >= 2 and error_rgb is not None and uncertainty_rgb is not None:
        #     canvas7 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)
        #     canvas7[:, 0:Wp * 2] = canvas1[:, 0:Wp * 2].copy()

        #     eval_utils.draw_text(
        #         canvas7, (20, Wp * 2 + 5), (0.5, 0.0),
        #         f'Proj+Error (PSNR Occ {frame_psnr_occ[0, t]:.2f}, SSIM Occ {frame_ssim_occ[0, t]:.2f})',
        #         (1, 1, 1), font_size)
        #     canvas7[40:Hp + 40, Wp * 2:Wp * 3] = reproject_rgb[t] * 0.4  # Darken reproj.
        #     canvas7[40:Hp + 40, Wp * 2:Wp * 3] += error_rgb[t] * 0.8

        #     eval_utils.draw_text(
        #         canvas7, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
        #         f'Proj+Uncert (Div {frame_diversity[t]:.3f})', (1, 1, 1), font_size)
        #     canvas7[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = reproject_rgb[t] * 0.4  # Darken reproj.
        #     canvas7[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] += uncertainty_rgb[t] * 0.8

        #     rich7_frames.append(canvas7)

    # Organize & return results.
    vis_dict = dict()

    # Pause a tiny bit at the beginning and end for less jerky looping.
    rich1_frames = [rich1_frames[0]] + rich1_frames + [rich1_frames[-1]] * 2
    rich1_frames = np.stack(rich1_frames, axis=0)
    rich1_frames = np.clip(rich1_frames, 0.0, 1.0)
    vis_dict['rich1'] = rich1_frames

    if len(rich2_frames) > 0:
        rich2_frames = [rich2_frames[0]] + rich2_frames + [rich2_frames[-1]] * 2
        rich2_frames = np.stack(rich2_frames, axis=0)
        rich2_frames = np.clip(rich2_frames, 0.0, 1.0)
        vis_dict['rich2'] = rich2_frames

    if len(rich3_frames) > 0:
        rich3_frames = [rich3_frames[0]] + rich3_frames + [rich3_frames[-1]] * 2
        rich3_frames = np.stack(rich3_frames, axis=0)
        rich3_frames = np.clip(rich3_frames, 0.0, 1.0)
        vis_dict['rich3'] = rich3_frames

    if len(rich4_frames) > 0:
        rich4_frames = [rich4_frames[0]] + rich4_frames + [rich4_frames[-1]] * 2
        rich4_frames = np.stack(rich4_frames, axis=0)
        rich4_frames = np.clip(rich4_frames, 0.0, 1.0)
        vis_dict['rich4'] = rich4_frames

    if len(rich5_frames) > 0:
        rich5_frames = [rich5_frames[0]] + rich5_frames + [rich5_frames[-1]] * 2
        rich5_frames = np.stack(rich5_frames, axis=0)
        rich5_frames = np.clip(rich5_frames, 0.0, 1.0)
        vis_dict['rich5'] = rich5_frames

    if len(rich6_frames) > 0:
        rich6_frames = [rich6_frames[0]] + rich6_frames + [rich6_frames[-1]] * 2
        rich6_frames = np.stack(rich6_frames, axis=0)
        rich6_frames = np.clip(rich6_frames, 0.0, 1.0)
        vis_dict['rich6'] = rich6_frames

    if len(rich7_frames) > 0:
        rich7_frames = [rich7_frames[0]] + rich7_frames + [rich7_frames[-1]] * 2
        rich7_frames = np.stack(rich7_frames, axis=0)
        rich7_frames = np.clip(rich7_frames, 0.0, 1.0)
        vis_dict['rich7'] = rich7_frames

    vis_dict['input'] = input_rgb
    vis_dict['gt'] = gt_rgb
    if reproject_rgb is not None:
        vis_dict['reproject'] = reproject_rgb
    vis_dict['output'] = pred_samples_rgb
    if error_rgb is not None:
        vis_dict['error'] = error_rgb
    if uncertainty_rgb is not None:
        vis_dict['uncertainty'] = uncertainty_rgb

    return vis_dict


def save_results(args, metrics_dict, vis_dict, controls, output_fp1, output_fp2):
    vis_fps = (6 + controls[2]) // 2
    eval_utils.write_video_and_frames(
        vis_dict['rich1'], dst_dp=output_fp1 + '_gal', fps=vis_fps,
        save_images=False, save_mp4=True, quality=9)

    if 'rich2' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich2'], dst_dp=output_fp1 + '_io', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich3' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich3'], dst_dp=output_fp1 + '_error', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich4' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich4'], dst_dp=output_fp1 + '_latent', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich5' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich5'], dst_dp=output_fp1 + '_uncert', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich6' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich6'], dst_dp=output_fp1 + '_visocc', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich7' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich7'], dst_dp=output_fp1 + '_projeu', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if args.save_images or args.save_mp4:
        if args.save_input:
            eval_utils.write_video_and_frames(
                vis_dict['input'], dst_dp=output_fp2 + '_input', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        if args.save_gt and 'gt' in vis_dict:
            eval_utils.write_video_and_frames(
                vis_dict['gt'], dst_dp=output_fp2 + '_gt', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        if args.reproject_rgbd and 'reproject' in vis_dict:
            eval_utils.write_video_and_frames(
                vis_dict['reproject'], dst_dp=output_fp2 + '_proj', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        for s in range(args.num_samples):
            eval_utils.write_video_and_frames(
                vis_dict['output'][s], dst_dp=output_fp2 + f'_pred_s{s}', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        if args.save_error and 'error' in vis_dict:
            eval_utils.write_video_and_frames(
                vis_dict['error'], dst_dp=output_fp2 + '_error', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        if args.save_uncertainty and 'uncertainty' in vis_dict:
            eval_utils.write_video_and_frames(
                vis_dict['uncertainty'], dst_dp=output_fp2 + '_uncert', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

    if args.calculate_metrics:
        eval_utils.save_json(metrics_dict, output_fp2 + '_metrics.json')


def process_example(args, worker_idx, example_idx, example, model_bundle, control_info):
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    # Load & preprocess input frames.
    print()
    print(f'[yellow]{worker_idx}: Loading input frames from {example}...')
    start_time = time.time()
    (input_rgb, gt_rgb, reproject_rgb, controls, batch) = load_input_gt(
        args, worker_idx, example, train_config, control_info, device)
    print(f'[magenta]{worker_idx}: Loading frames took {time.time() - start_time:.2f}s')

    # Run inference.
    print()
    print(f'[cyan]{worker_idx}: Running SVD model on selected video clip...')
    start_time = time.time()
    if args.num_samples >= 1:
        pred_samples = run_inference(
            args, device, model, batch)
    else:
        pred_samples = []
    print(f'[magenta]{worker_idx}: Inference took {time.time() - start_time:.2f}s')

    # Calculate metrics.
    if args.calculate_metrics and args.num_samples >= 1:
        print()
        print(f'[cyan]{worker_idx}: Calculating metrics...')
        start_time = time.time()
        (metrics_dict, uncertainty) = calculate_metrics(
            args, gt_rgb, reproject_rgb, pred_samples)
        print(f'[magenta]{worker_idx}: Metrics took {time.time() - start_time:.2f}s')
    else:
        metrics_dict = dict()
        uncertainty = None

    # Create rich inference visualization.
    print()
    print(f'[yellow]{worker_idx}: Creating rich visualizations...')
    start_time = time.time()
    controls_friendly = get_controls_friendly(controls)
    vis_dict = create_visualizations(
        args, input_rgb, gt_rgb, reproject_rgb, controls_friendly, pred_samples,
        metrics_dict, uncertainty, model_name)
    print(f'[magenta]{worker_idx}: Visualizations took {time.time() - start_time:.2f}s')

    # Prepare output directory.
    test_tag = os.path.basename(args.output).split('_')[0]
    output_fn = os.path.splitext(os.path.basename(example))[0]
    output_fn = output_fn.replace('_p0', '')
    output_fn = output_fn.replace('_rgb', '')

    output_fn1 = f'{test_tag}_{example_idx:03d}_n{model_name}'
    if control_info is not None:
        output_fn1 += f'_c{args.control_idx:02d}'
    output_fn1 += f'_{output_fn}'
    output_fn2 = output_fn1  # Save shorter name for extra data.

    # Contains either just frame rate or frame rate + camera controls.
    output_fn1 += f'_{controls_friendly[2]}'

    # if args.num_samples >= 1:
    #     psnr = np.nanmean(metrics_dict['mean_psnr'])
    #     ssim = np.nanmean(metrics_dict['mean_ssim'])
    #     diversity = metrics_dict['mean_diversity']
    #     # output_fn1 += f'_psnr{psnr:.2f}_ssim{ssim:.3f}_div{diversity:.3f}'

    # For more prominent / visible stuff:
    output_fp1 = os.path.join(args.output, output_fn1)

    # For less prominent but still useful stuff:
    output_fp2 = os.path.join(args.output, 'extra', output_fn2)

    # Save results to disk.
    print()
    print(f'[yellow]{worker_idx}: Saving results to disk...')
    start_time = time.time()
    save_results(args, metrics_dict, vis_dict, controls, output_fp1, output_fp2)
    print(f'[magenta]{worker_idx}: Saving took {time.time() - start_time:.2f}s')

    return True


def worker_fn(args, worker_idx, num_workers, gpu_idx, model_path, example_list):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    # Only now can we import torch.
    import torch
    from sgm.util import instantiate_from_config
    torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)

    # Update CPU affinity.
    eval_utils.update_os_cpu_affinity(worker_idx, num_workers)

    if not (os.path.exists(model_path)) and '*' in model_path:
        used_model_path = sorted(glob.glob(model_path))[-1]
        print(f'[orange3]{worker_idx}: Warning: Parsed {model_path} '
              f'to assumed latest checkpoint {used_model_path}')
    else:
        used_model_path = model_path

    print()
    print(f'[cyan]{worker_idx}: Loading SVD model from {used_model_path} on GPU {gpu_idx}...')
    start_time = time.time()

    device = args.device
    if 'cuda' in device:
        device = f'cuda:{gpu_idx}'

    # Initialize model.
    model_bundle = eval_utils.load_model_bundle(
        device, args.config_path, used_model_path, args.use_ema,
        num_steps=args.num_steps, num_frames=args.num_frames,
        max_scale=args.guider_max_scale, min_scale=args.guider_min_scale,
        verbose=(worker_idx == 0))
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    print(f'[magenta]{worker_idx}: Loading SVD model took {time.time() - start_time:.2f}s')

    eval_utils.warn_resolution_mismatch(train_config, args.frame_width, args.frame_height)

    # Load control information for consistent quantitative evaluation.
    if len(args.control_json) > 0:
        control_info = eval_utils.load_json(args.control_json)
    else:
        control_info = None

    # Start iterating over all videos.
    if args.debug:
        to_loop = tqdm.tqdm(list(enumerate(example_list)))
    else:
        to_loop = tqdm.rich.tqdm(list(enumerate(example_list)))

    # Enable EMA scope early on to avoid constant shifting around of weights.
    with model.ema_scope('Testing'):

        for (i, example) in to_loop:
            example_idx = i * num_workers + worker_idx

            try:
                process_example(args, worker_idx, example_idx, example, model_bundle, control_info)

            except Exception as e:
                print(f'[red]{worker_idx}: Error processing {example}: {e}')
                print(f'[red]Traceback: {traceback.format_exc()}')
                print(f'[red]Skipping...')
                continue

    print()
    print(f'[cyan]{worker_idx}: Done!')
    print()


def main(args):

    # Save the arguments to this training script.
    args_fp = os.path.join(args.output, 'args_test.json')
    eval_utils.save_json(vars(args), args_fp)
    print(f'[yellow]Saved script args to {args_fp}')

    # Load list of videos to process (not the pixels themselves yet).
    print()
    print(f'[yellow]Parsing list of individual examples from {args.input}...')
    start_time = time.time()

    examples = eval_utils.get_list_of_input_images_or_videos(args.input)
    print(f'[yellow]Found {len(examples)} examples '
          f'(counting both video files and/or image folders).')

    print(f'[magenta]Loading data list took {time.time() - start_time:.2f}s')

    assert len(examples) > 0, f'No examples found in {args.input}!'

    # Split models and examples across workers.
    assert len(args.gpus) % len(args.model_path) == 0, \
        f'Number of GPUs ({args.gpus}) must be a multiple of number of models ({args.model_path})!'
    num_gpus = len(args.gpus)
    num_models = len(args.model_path)
    num_buckets = num_gpus // num_models
    worker_args_list = []

    for worker_idx in range(num_gpus):
        # Every model must see every example once in total,
        # but every GPU per model must see strictly different subsets of examples.
        gpu_idx = args.gpus[worker_idx % num_gpus]
        model_idx = worker_idx % num_models
        bucket_idx = worker_idx // num_models

        model_path = args.model_path[model_idx]
        my_examples = examples[bucket_idx::num_buckets]
        worker_args_list.append((args, worker_idx, num_gpus, gpu_idx,
                                 model_path, my_examples))

    print(f'[cyan]Splitting {len(examples)} examples across {num_gpus} workers '
          f'for {num_models} models according to specified GPU devices: {args.gpus}...')
    start_time = time.time()

    if num_gpus > 1:
        print(f'[cyan]Starting {num_gpus} processes...')

        # https://github.com/pytorch/pytorch/issues/40403
        import torch
        import torch.multiprocessing
        torch.multiprocessing.set_start_method('spawn')

        with mp.Pool(processes=num_gpus) as pool:
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

    args = test_args()

    main(args)

    pass
