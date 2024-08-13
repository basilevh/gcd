'''
Created by Basile Van Hoorick for GCD, 2024.
Kubric-4D data loader (requires some data preprocessing).
'''

import os  # noqa
import sys  # noqa

# Library imports.
import lovely_tensors
import multiprocessing as mp
import numpy as np
import pytorch_lightning as pl
import time
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data
import traceback
from einops import rearrange
from lovely_numpy import lo
from rich import print

# Internal imports.
from sgm.data import common
from sgm.data import geometry

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)


class KubricSynthViewDataset(torch.utils.data.Dataset):

    def __init__(
            self, dset_root, start_idx, end_idx, force_shuffle=False,
            pcl_root='',
            avail_frames=60, model_frames=14,
            input_frames=7, output_frames=14,
            center_crop=True, frame_width=384, frame_height=256,
            input_mode='arbitrary', output_mode='arbitrary',
            azimuth_range=[0.0, 360.0],
            elevation_range=[0.0, 50.0],
            radius_range=[12.0, 18.0],
            delta_azimuth_range=[-60.0, 60.0],
            delta_elevation_range=[-30.0, 30.0],
            delta_radius_range=[-3.0, 3.0],
            elevation_sample_sin=False,
            trajectory='interpol_linear', move_time=10,
            camera_control='spherical', motion_bucket_range=[127, 127],
            cond_aug=0.02, mock_dset_size=1000,
            reverse_prob=0.2, data_gpu=0,
            spread_radius=1, render_width=420, render_height=280,
            **kwargs):
        super().__init__()
        self.dset_root = dset_root
        self.pcl_root = pcl_root
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_scenes = end_idx - start_idx
        self.force_shuffle = force_shuffle
        self.avail_frames = avail_frames
        self.model_frames = model_frames
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.center_crop = center_crop
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.radius_range = radius_range
        self.delta_azimuth_range = delta_azimuth_range
        self.delta_elevation_range = delta_elevation_range
        self.delta_radius_range = delta_radius_range
        self.elevation_sample_sin = elevation_sample_sin
        self.trajectory = trajectory
        self.move_time = move_time
        self.camera_control = camera_control
        self.motion_bucket_range = motion_bucket_range
        self.cond_aug = cond_aug
        self.mock_dset_size = mock_dset_size
        self.reverse_prob = reverse_prob
        self.data_gpu = data_gpu
        self.spread_radius = spread_radius
        self.render_width = render_width
        self.render_height = render_height

        self.avail_views = 16
        self.avail_frames = min(self.avail_frames, 60)
        self.avail_fps = 24

        self.next_example = None
        self.total_counter = mp.Value('i', 0)
        self.max_retries = 100
        self.reproject_rgbd = False

    def set_next_example(self, *args):
        '''
        For evaluation purposes.
        '''
        # Typically, args = [scene_idx, frame_skip, frame_start, reverse,
        # azimuth_start, azimuth_end, elevation_start, elevation_end, radius_start, radius_end].
        self.next_example = [*args]

    def __len__(self):
        return self.mock_dset_size

    def __getitem__(self, idx):
        verbose = (self.total_counter.value <= 10 or self.total_counter.value % 200 == 0)
        self.total_counter.value += 1

        start_time = time.time()

        Tv = self.avail_frames
        Tcm = self.model_frames

        scene_idx = -1
        scene_dn = ''
        scene_dp = ''

        # We have to fend ourselves against incomplete data conversions / exports.
        for retry_idx in range(self.max_retries):
            try:
                if self.next_example is not None:
                    print(f'[cyan]Loading next_example: {self.next_example}')
                    scene_idx = int(self.next_example[0])
                    frame_skip = int(self.next_example[1])
                    frame_start = int(self.next_example[2])
                    reverse = bool(self.next_example[3])

                else:
                    if retry_idx >= 1 or self.force_shuffle:
                        # Try again with a random index.
                        idx2 = np.random.randint(0, self.mock_dset_size)
                        idx = (idx2 + idx) % self.mock_dset_size
                    scene_idx = idx % self.num_scenes + self.start_idx

                    # Choose a random speed for each clip subsampled from the video.
                    max_skip = Tv // Tcm  # Floor division; usually = 4.
                    frame_skip = np.random.randint(1, max_skip + 1)

                    # Also allow for a modest random offset to avoid always starting at the same frame.
                    desired_max_offset = 6
                    cover_video = frame_skip * (Tcm - 1) + 1  # Number of frames; inclusive.
                    max_frame_start = Tv - cover_video - 1  # Highest possible offset; inclusive.
                    used_max_frame_start = max(min(max_frame_start, desired_max_offset), 0)
                    frame_start = np.random.randint(0, used_max_frame_start + 1)

                    # Apply random temporal data augmentations.
                    reverse = (np.random.rand() < self.reverse_prob)

                scene_dp = os.path.join(self.dset_root, f'scn{scene_idx:05d}')
                scene_dn = os.path.basename(scene_dp)
                pcl_dp = os.path.join(self.pcl_root, f'scn{scene_idx:05d}')

                # Obtain resulting final clip frame indices.
                fps = int(round(self.avail_fps / frame_skip))  # Becomes 24 / 12 / 8 / 6.
                clip_frames = np.arange(Tcm) * frame_skip + frame_start
                if scene_idx >= 0:
                    assert 0 <= clip_frames[0] and clip_frames[-1] <= Tv - 1, str(clip_frames)

                if reverse:
                    clip_frames = clip_frames[::-1].copy()

                # Load scene metadata and rendered viewpoint camera matrices.
                # NOTE: We actually load the first "dense low down" viewpoint in kubmv10,
                # because it should match extrinsics_src in kubric_valtest_controls JSON exactly.
                if scene_idx >= 0:
                    metadata_fp = os.path.join(scene_dp, f'{scene_dn}_p0_v4.json')
                    metadata = common.load_json(metadata_fp)
                    (first_intrinsics, first_extrinsics) = \
                        geometry.get_kubric_camera_matrices_torch(metadata)
                    # (T, 3, 3) tensor of float32; (T, 4, 4) tensor of float32.
                else:
                    metadata = None

                # Load 3D point cloud video clip (requires dataset conversion beforehand!).
                if scene_idx >= 0:
                    pcl_dict = self.load_point_clouds(
                        pcl_dp, clip_frames, verbose)
                else:
                    pcl_dict = None

                # Define input and output camera trajectories.
                (spherical_start, spherical_end, spherical_src, spherical_dst,
                 extrinsics_src, extrinsics_dst, motion_amount) = \
                    self.sample_trajectories(verbose)
                # 2x (Tcm, 3) tensor of float32; 2x (Tcm, 4, 4) tensor of float32;

                # Synthesize input and target viewpoint videos.
                if scene_idx >= 0:
                    with torch.no_grad():
                        (rgb_src, rgb_dst, reproject) = self.synth_src_dst_rgb(
                            pcl_dict, extrinsics_src, extrinsics_dst,
                            first_intrinsics, first_extrinsics)
                    # 3x (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
                else:
                    rgb_src = None
                    rgb_dst = None
                    reproject = None

                # Now construct the final tensors that will be actually used in the model pipeline.
                data_dict = self.construct_dict(
                    rgb_src, rgb_dst, reproject, fps, spherical_src, spherical_dst,
                    extrinsics_src, extrinsics_dst, motion_amount, verbose)

                # We successfully loaded this example if we reach this.
                break

            except Exception as e:
                wait_time = 0.2 + retry_idx * 0.02
                if verbose or retry_idx in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
                    print(f'[red]Warning: Skipping example that failed to load: '
                          f'scene_idx: {scene_idx}  scene_dn: {scene_dp}  '
                          f'exception: {e}  retry_idx: {retry_idx}  wait_time: {wait_time:.2f}')
                if verbose and retry_idx == 4:
                    print(f'[red]Traceback: {traceback.format_exc()}')
                if retry_idx >= self.max_retries - 2:
                    raise e  # This effectively stops the training job.
                time.sleep(wait_time)

        # Add extra info / metadata for debugging / logging.
        data_dict['dset'] = torch.tensor([1])
        data_dict['idx'] = torch.tensor([idx])
        data_dict['scene_idx'] = torch.tensor([scene_idx])
        data_dict['frame_start'] = torch.tensor([frame_start])
        data_dict['frame_skip'] = torch.tensor([frame_skip])
        data_dict['clip_frames'] = torch.tensor(clip_frames)

        if verbose:
            print(f'[gray]KubricSynthViewDataset __getitem__ '
                  f'idx: {idx} scene_idx: {scene_idx} took: {time.time() - start_time:.3f} s')

        return data_dict

    def load_point_clouds(self, pcl_dp, clip_frames, verbose):
        all_xyz = []
        all_rgb = []
        all_segm_rgb = []

        for t in clip_frames:
            pcl_fp = os.path.join(pcl_dp, f'pcl_rgb_segm_{t:05d}.pt')
            pcl_all = torch.load(pcl_fp)

            (pcl_xyz, pcl_rgb, pcl_segm_rgb) = pcl_all
            # (V, N, 3) of float16; (V, N, 3) of uint8; (V, N, 3) of uint8.

            all_xyz.append(pcl_xyz)
            all_rgb.append(pcl_rgb)
            all_segm_rgb.append(pcl_segm_rgb)

        # Do not stack, but keep lists, for efficiency.
        pcl_dict = dict()
        pcl_dict['xyz'] = all_xyz  # List of (V, N, 3) tensor of float16.
        pcl_dict['rgb'] = all_rgb  # List of (V, N, 3) tensor of uint8.
        pcl_dict['segm_rgb'] = all_segm_rgb  # List of (V, N, 3) tensor of uint8.

        return pcl_dict

    def sample_trajectories(self, verbose, spherical_start=None, spherical_end=None):
        '''
        :return spherical_start: (3) tensor of float32.
        :return spherical_end: (3) tensor of float32.
        :return spherical_src: (Tcm, 3) tensor of float32.
        :return spherical_dst: (Tcm, 3) tensor of float32.
        :return extrinsics_src: (Tcm, 4, 4) tensor of float32.
        :return extrinsics_dst: (Tcm, 4, 4) tensor of float32.
        :return motion_amount: float32 in [0, 1].
        '''
        Tcm = self.model_frames

        assert self.input_mode == 'arbitrary'
        assert self.output_mode == 'arbitrary'

        if self.next_example is not None and self.next_example[4] > -1000:
            azimuth_start = float(self.next_example[4])
            azimuth_end = float(self.next_example[5])
            elevation_start = float(self.next_example[6])
            elevation_end = float(self.next_example[7])
            radius_start = float(self.next_example[8])
            radius_end = float(self.next_example[9])

        else:
            if spherical_start is None:
                if self.azimuth_range[1] - self.azimuth_range[0] <= 0.0:
                    azimuth_start = self.azimuth_range[0]
                else:
                    azimuth_start = np.random.uniform(*self.azimuth_range)

                if self.elevation_range[1] - self.elevation_range[0] <= 0.0:
                    elevation_start = self.elevation_range[0]
                else:
                    if self.elevation_sample_sin:
                        # This ensures better coverage of the unit sphere because it makes large
                        # elevations (in both directions) less likely.
                        elev_bounds = np.array([np.sin(self.elevation_range[0] / 180.0 * np.pi),
                                                np.sin(self.elevation_range[1] / 180.0 * np.pi)])
                        sin_elev_sample = np.random.uniform(*elev_bounds)
                        elevation_start = np.arcsin(sin_elev_sample) * 180.0 / np.pi
                        # if verbose:
                        #     print(f'[gray]elev_bounds: {elev_bounds}  '
                        #           f'sin_elev_sample: {sin_elev_sample:.3f}  '
                        #           f'elevation_start: {elevation_start:.3f}')
                    else:
                        # This may end up being too concentrated towards the two poles.
                        elevation_start = np.random.uniform(*self.elevation_range)

                if self.radius_range[1] - self.radius_range[0] <= 0.0:
                    radius_start = self.radius_range[0]
                else:
                    radius_start = np.random.uniform(*self.radius_range)

            else:
                azimuth_start = spherical_start[0]
                elevation_start = spherical_start[1]
                radius_start = spherical_start[2]

            if spherical_end is None:
                if self.delta_azimuth_range[1] - self.delta_azimuth_range[0] <= 0.0:
                    # NOTE: This fixed offset overrides the range check!
                    azimuth_end = azimuth_start + self.delta_azimuth_range[0]
                elif self.azimuth_range[1] - self.azimuth_range[0] >= 360.0:
                    azimuth_end = azimuth_start + np.random.uniform(*self.delta_azimuth_range)
                else:
                    azimuth_end = np.random.uniform(
                        max(azimuth_start + self.delta_azimuth_range[0], self.azimuth_range[0]),
                        min(azimuth_start + self.delta_azimuth_range[1], self.azimuth_range[1]))

                if len(self.delta_elevation_range) != 2:
                    # NOTE: This is an single ABSOLUTE value, rather than a (possible variable) offset!
                    elevation_end = self.delta_elevation_range[0]
                elif self.delta_elevation_range[1] - self.delta_elevation_range[0] <= 0.0:
                    # NOTE: This fixed offset overrides the range check!
                    elevation_end = elevation_start + self.delta_elevation_range[0]
                else:
                    elevation_end = np.random.uniform(
                        max(elevation_start +
                            self.delta_elevation_range[0], self.elevation_range[0]),
                        min(elevation_start + self.delta_elevation_range[1], self.elevation_range[1]))

                if len(self.delta_radius_range) != 2:
                    # NOTE: This is an single ABSOLUTE value, rather than a (possible variable) offset!
                    radius_end = self.delta_radius_range[0]
                elif self.delta_radius_range[1] - self.delta_radius_range[0] <= 0.0:
                    # NOTE: This fixed offset overrides the range check!
                    radius_end = radius_start + self.delta_radius_range[0]
                else:
                    radius_end = np.random.uniform(
                        max(radius_start + self.delta_radius_range[0], self.radius_range[0]),
                        min(radius_start + self.delta_radius_range[1], self.radius_range[1]))

            else:
                azimuth_end = spherical_end[0]
                elevation_end = spherical_end[1]
                radius_end = spherical_end[2]

        spherical_start = np.array([azimuth_start, elevation_start, radius_start],
                                   dtype=np.float32)
        spherical_end = np.array([azimuth_end, elevation_end, radius_end],
                                 dtype=np.float32)

        if verbose:
            print(f'[gray]spherical_start: {spherical_start}  spherical_end: {spherical_end}')
        # 2x (3) array of float32.

        my_motion = np.linalg.norm(spherical_end[0:2] - spherical_start[0:2], ord=2)
        max_motion = np.linalg.norm([max(*self.delta_azimuth_range),
                                     max(*self.delta_elevation_range)], ord=2)
        motion_amount = my_motion / max_motion

        (spherical_src, spherical_dst) = common.construct_trajectory(
            spherical_start, spherical_end, self.trajectory, Tcm, self.move_time)
        # 2x (Tcm, 3) array of float32.

        # Determine viewing directions by pair of positions over time.
        # NOTE: It is important to add same Z-offset here as look_at to
        # ensure spherical rotations are consistent and meaningful!
        position_src = geometry.cartesian_from_spherical(spherical_src, deg2rad=True)
        position_src[..., 2] += 1.0
        position_dst = geometry.cartesian_from_spherical(spherical_dst, deg2rad=True)
        position_dst[..., 2] += 1.0
        # 2x (Tcm, 3) array of float32.

        look_at_src = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        look_at_dst = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        look_at_src = np.tile(look_at_src[None], (Tcm, 1))
        look_at_dst = np.tile(look_at_dst[None], (Tcm, 1))
        # 2x (Tcm, 3) array of float32.

        # Convert all to camera extrinsics over time.
        extrinsics_src = np.zeros((Tcm, 4, 4), dtype=np.float32)
        extrinsics_dst = np.zeros((Tcm, 4, 4), dtype=np.float32)
        for t in range(0, Tcm):
            extrinsics_src[t] = geometry.extrinsics_from_look_at(position_src[t], look_at_src[t])
            extrinsics_dst[t] = geometry.extrinsics_from_look_at(position_dst[t], look_at_dst[t])

        # Convert all to pytorch tensors.
        spherical_start = torch.tensor(spherical_start, dtype=torch.float32)
        spherical_end = torch.tensor(spherical_end, dtype=torch.float32)
        spherical_src = torch.tensor(spherical_src, dtype=torch.float32)
        spherical_dst = torch.tensor(spherical_dst, dtype=torch.float32)
        extrinsics_src = torch.tensor(extrinsics_src, dtype=torch.float32)
        extrinsics_dst = torch.tensor(extrinsics_dst, dtype=torch.float32)

        return (spherical_start, spherical_end, spherical_src, spherical_dst,
                extrinsics_src, extrinsics_dst, motion_amount)

    def synth_src_dst_rgb(self, pcl_dict, extrinsics_src, extrinsics_dst,
                          avail_intrinsics, avail_extrinsics):
        '''
        :param pcl_dict: dict.
        :param extrinsics_src: (Tcm, 4, 4) tensor of float32.
        :param extrinsics_dst: (Tcm, 4, 4) tensor of float32.
        :param avail_intrinsics: (Tcm, 3, 3) tensor of float32.
        :param avail_extrinsics: (Tcm, 4, 4) tensor of float32.
        :return rgb_src: (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
        :return rgb_dst: (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
        :return reproject: Optional (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
        '''
        Tcm = self.model_frames
        blur_radius = 21
        reproject_blur_radius = 3  # Much smaller because this is for baseline.

        # We dedicate a special GPU for point cloud processing and rendering.
        # NOTE: This takes up quite a bit of VRAM so should be separate from all the model training!
        device = f'cuda:{self.data_gpu}'

        pcl_xyz = pcl_dict['xyz']  # List of (V, N, 3) tensor of float16.
        pcl_rgb = pcl_dict['rgb']  # List of (V, N, 3) tensor of uint8.
        pcl_segm_rgb = pcl_dict['segm_rgb']  # List of (V, N, 3) tensor of uint8.

        # NOTE: In Kubric, the instrinsics are with respect to normalized image space [0, 1]^2,
        # but project_points_to_pixels() expects unnormalized pixel space, so correct here.
        used_intrinsics = avail_intrinsics[0].clone()
        used_intrinsics[0, :] *= self.render_width
        used_intrinsics[1, :] *= self.render_height

        # We also have to correct the aspect ratio, which corresponds to cropping (in the exact same
        # manner as the earlier frame loading pipeline) to avoid stretching!
        old_ar = 576.0 / 384.0
        new_ar = self.render_width / self.render_height
        if new_ar > old_ar + 1e-3:
            # Because of how camera projection works, both operations result in equivalent outcomes:
            # used_intrinsics[:, 1, 1] *= new_ar / old_ar
            used_intrinsics[1, 1] = used_intrinsics[0, 0]
        elif new_ar < old_ar - 1e-3:
            used_intrinsics[0, 0] = used_intrinsics[1, 1]

        rgb_src = []
        rgb_dst = []
        if self.reproject_rgbd:
            reproject = []
        else:
            reproject = None

        for t in range(Tcm):
            cur_xyz = pcl_xyz[t].type(torch.float32)  # (V, N, 3) tensor of float32.
            cur_rgb = (pcl_rgb[t] / 255.0).type(torch.float32)  # (V, N, 3) tensor of float32.
            cur_xyz = cur_xyz.to(device)
            cur_rgb = cur_rgb.to(device)
            cur_xyzrgb = torch.cat([cur_xyz, cur_rgb], axis=-1)  # (V, N, 6) tensor of float32.
            del cur_xyz
            del cur_rgb

            cur_xyzrgb = cur_xyzrgb.reshape(-1, 6)  # (V * N, 6) tensor of float32.
            cur_xyzrgb = cur_xyzrgb.type(torch.float64)  # (V * N, 6) tensor of float64.

            # This occurs at rendering resolution, which is ideally inbetween dataset and model.
            (cur_synth1, cur_weights, cur_uv, cur_depth) = geometry.project_points_to_pixels(
                cur_xyzrgb, used_intrinsics, extrinsics_src[t],
                self.render_height, self.render_width, spread_radius=self.spread_radius)
            # (H, W, 3) tensor of float32 in [0, 1];
            # (H, W, 1) tensor of float64 in [0, 1];
            # (N, 2) tensor of float64;
            # (H, W, 1) tensor of float64.
            blur_synth1 = geometry.blur_into_black(
                cur_synth1, kernel_size=blur_radius, sigma=blur_radius / 4.0)
            # (H, W, 3) tensor of float32 in [0, 1].

            # Resize to final model resolution.
            blur_synth1 = rearrange(blur_synth1, 'h w c -> c h w')
            blur_synth1 = torch.nn.functional.interpolate(
                blur_synth1[None], (self.frame_height, self.frame_width), mode='bilinear',
                align_corners=False)[0]
            # (3, H, W) tensor of float32 in [0, 1].

            blur_synth1 = blur_synth1.cpu()
            rgb_src.append(blur_synth1)
            del cur_synth1

            (cur_synth2, cur_weights, cur_uv, cur_depth) = geometry.project_points_to_pixels(
                cur_xyzrgb, used_intrinsics, extrinsics_dst[t],
                self.render_height, self.render_width, spread_radius=self.spread_radius)
            blur_synth2 = geometry.blur_into_black(
                cur_synth2, kernel_size=blur_radius, sigma=blur_radius / 4.0)
            blur_synth2 = rearrange(blur_synth2, 'h w c -> c h w')
            blur_synth2 = torch.nn.functional.interpolate(
                blur_synth2[None], (self.frame_height, self.frame_width), mode='bilinear',
                align_corners=False)[0]
            # (3, H, W) tensor of float32 in [0, 1].

            blur_synth2 = blur_synth2.cpu()
            rgb_dst.append(blur_synth2)
            del cur_synth2

            if reproject is not None:
                print(f'[cyan]Reprojecting source RGBD to destination viewpoint...')
                input_extrinsics = extrinsics_src[t]
                match_extrinsics = avail_extrinsics[t]

                # We loaded the first "dense low down" viewpoint in kubmv10, so first verify that
                # we are actually using the correct control trajectories for testing.
                mismatch = (input_extrinsics - match_extrinsics).abs().sum()
                if mismatch > 0.005:
                    print(f'[bold][red]=> WARNING: Mismatch in available versus '
                          f'used input extrinsics! {mismatch:.6f}')
                else:
                    print(f'[green]Available and used input extrinsics are aligned! '
                          f'{mismatch:.6f}')

                # NOTE: Hardcoded viewpoint index value for stored point cloud data.
                src_xyz = pcl_xyz[t][4].type(torch.float32).to(device)
                src_rgb = (pcl_rgb[t][4] / 255.0).type(torch.float32).to(device)
                src_xyzrgb = torch.cat([src_xyz, src_rgb], axis=-1)  # (N, 6) tensor of float32.
                del src_xyz
                del src_rgb

                src_xyzrgb = src_xyzrgb.type(torch.float64)  # (N, 6) tensor of float64.
                (cur_synth3, cur_weights, cur_uv, cur_depth) = geometry.project_points_to_pixels(
                    src_xyzrgb, used_intrinsics, extrinsics_dst[t],
                    self.render_height, self.render_width, spread_radius=self.spread_radius)
                blur_synth3 = geometry.blur_into_black(
                    cur_synth3, kernel_size=reproject_blur_radius,
                    sigma=reproject_blur_radius / 4.0)
                blur_synth3 = rearrange(blur_synth3, 'h w c -> c h w')
                blur_synth3 = torch.nn.functional.interpolate(
                    blur_synth3[None], (self.frame_height, self.frame_width), mode='bilinear',
                    align_corners=False)[0]
                # (3, H, W) tensor of float32 in [0, 1].

                blur_synth3 = blur_synth3.cpu()
                reproject.append(blur_synth3)
                del cur_synth3

            del cur_xyzrgb
            del cur_weights
            del cur_uv
            del cur_depth

        rgb_src = torch.stack(rgb_src, dim=0)
        rgb_src = rgb_src * 2.0 - 1.0
        # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
        rgb_dst = torch.stack(rgb_dst, dim=0)
        rgb_dst = rgb_dst * 2.0 - 1.0
        # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
        if reproject is not None:
            reproject = torch.stack(reproject, dim=0)
            reproject = reproject * 2.0 - 1.0
            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

        return (rgb_src, rgb_dst, reproject)

    def construct_dict(self, rgb_src, rgb_dst, reproject, fps, spherical_src, spherical_dst,
                       extrinsics_src, extrinsics_dst, motion_amount, verbose):
        Tcm = self.model_frames
        Tci = self.input_frames
        Tco = self.output_frames

        cond_aug = torch.ones((Tcm,), dtype=torch.float32) * self.cond_aug
        # (Tcm) = (14) tensor of float32 = all 0.02.

        # Assign appropriate motion value if model is being trained with synchronized values.
        motion_range = self.motion_bucket_range[1] - self.motion_bucket_range[0]
        if motion_range <= 0:
            motion_value = int(self.motion_bucket_range[0])
        else:
            motion_value = int(round(self.motion_bucket_range[0] +
                                     motion_range * motion_amount))
        # if verbose:
        #     print(f'[gray]motion_amount: {motion_amount:.3f}  motion_value: {motion_value}')
        motion_bucket_id = torch.ones((Tcm,), dtype=torch.int32) * motion_value
        # (Tcm) = (14) tensor of int32 = all 127.

        fps_id = torch.ones((Tcm,), dtype=torch.int32) * fps
        # (Tcm) = (14) tensor of int32 = all 6 / 8 / 12 / 24.
        image_only_indicator = torch.zeros((1, Tcm), dtype=torch.float32)
        # (1, Tcm) = (1, 14) tensor of float32 = all 0.

        # May be used by CameraEmbedder:
        scaled_rel_pose = torch.zeros((Tcm, 3, 4), dtype=torch.float32)
        for t in range(Tcm):
            RT1 = extrinsics_src[t]
            RT2 = extrinsics_dst[t]
            delta_RT = torch.linalg.inv(RT1) @ RT2
            scaled_rel_pose[t] = delta_RT[0:3, 0:4]  # .flatten()
        # (Tcm, 12) tensor of float32.

        # May be used by SphericalEmbedder:
        scaled_rel_angles = spherical_dst - spherical_src
        scaled_rel_angles[:, 0] *= torch.pi / 180.0
        scaled_rel_angles[:, 1] *= torch.pi / 180.0
        # (Tcm, 3) tensor of float32.

        data_dict = dict()
        data_dict['cond_aug'] = cond_aug.type(torch.float32)
        data_dict['motion_bucket_id'] = motion_bucket_id.type(torch.int32)
        data_dict['fps_id'] = fps_id.type(torch.int32)
        data_dict['image_only_indicator'] = image_only_indicator.type(torch.float32)
        data_dict['scaled_relative_pose'] = scaled_rel_pose.type(torch.float32)
        data_dict['scaled_relative_angles'] = scaled_rel_angles.type(torch.float32)

        if rgb_src is not None and rgb_dst is not None:
            target_frames = rgb_dst
            if Tco < Tcm:
                target_frames = torch.cat(
                    [target_frames[0:Tco]] +
                    [target_frames[Tco - 1:Tco]] * (Tcm - Tco), dim=0)
            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

            cond_frames_without_noise = rgb_src
            if Tci < Tcm:
                cond_frames_without_noise = torch.cat(
                    [cond_frames_without_noise[0:Tci]] +
                    [cond_frames_without_noise[Tci - 1:Tci]] * (Tcm - Tci), dim=0)
            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

            assert target_frames.shape[-2:] == (self.frame_height, self.frame_width), \
                f'{target_frames.shape} vs {self.frame_height} x {self.frame_width}'
            assert target_frames.shape == cond_frames_without_noise.shape, \
                f'{target_frames.shape} vs {cond_frames_without_noise.shape}'

            # Obtain used conditioning signal by adding noise to the input frames.
            cond_frames = (cond_frames_without_noise
                           + self.cond_aug * torch.randn(*cond_frames_without_noise.shape))
            # (Tcm, 3, Hp, Wp) tensor of float32 in [~-1, ~1].

            data_dict['jpg'] = target_frames.type(torch.float32)
            data_dict['cond_frames'] = cond_frames.type(torch.float32)
            data_dict['cond_frames_without_noise'] = cond_frames_without_noise.type(torch.float32)

        if reproject is not None:
            data_dict['reproject'] = reproject.type(torch.float32)
            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

        return data_dict


def collate_fn(example_list):
    collated = torch.utils.data.default_collate(example_list)
    # Correct result by merging batch & temporal dimensions.
    batch = {k: rearrange(v, 'b t ... -> (b t) ...') for (k, v) in collated.items()}
    batch['num_video_frames'] = batch['image_only_indicator'].shape[-1]
    return batch


class KubricSynthViewModule(pl.LightningDataModule):

    def __init__(
            self, dset_root, train_videos, val_videos, test_videos,
            batch_size, num_workers, shuffle=True, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = KubricSynthViewDataset(
            dset_root, 0, train_videos, **kwargs)
        self.val_dataset = KubricSynthViewDataset(
            dset_root, train_videos, train_videos + val_videos, **kwargs)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
