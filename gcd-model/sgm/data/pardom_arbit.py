'''
Created by Basile Van Hoorick for GCD, 2024.
ParallelDomain-4D data loader (requires some data preprocessing).
'''

import os  # noqa
import sys  # noqa

# Library imports.
import glob
import json
import lovely_tensors
import multiprocessing as mp
import numpy as np
import pathlib
import pytorch_lightning as pl
import sys
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


class ParallelDomainSynthViewDataset(torch.utils.data.Dataset):

    def __init__(
            self, dset_root, split, start_idx, end_idx, force_shuffle=False,
            pcl_root='', split_json='',
            avail_frames=50, model_frames=14,
            input_frames=7, output_frames=14,
            center_crop=True, frame_width=384, frame_height=256,
            input_mode='ego_forward', output_mode='topdown1',
            input_modality='rgb', output_modality='rgb',
            dst_cam_position=[-8.0, 0.0, 8.0],
            dst_cam_look_at=[5.60, 0.0, 1.55],
            dst_azimuth_range=[0.0, 0.0],
            dst_forward_offset=8.0,  # When azimuth is +/- 90 deg, position & look_at x are this.
            dst_pos_side_offset=9.0,  # When azimuth is +/- 90 deg, position y is +/- this.
            dst_look_side_offset=-1.20,  # When azimuth is +/- 90 deg, look_at y is +/- this.
            trajectory='interpol_sine', move_time=10, modal_time=0,
            camera_control='none', motion_bucket_range=[127, 127],
            cond_aug=0.02, mock_dset_size=1000,
            reverse_prob=0.05, data_gpu=0,
            spread_radius=1, render_width=420, render_height=280,
            **kwargs):
        super().__init__()
        self.dset_root = dset_root
        self.pcl_root = pcl_root
        self.split = split
        self.force_shuffle = force_shuffle
        self.split_json = split_json
        self.avail_frames = avail_frames
        self.model_frames = model_frames
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.center_crop = center_crop
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.dst_cam_position = dst_cam_position
        self.dst_cam_look_at = dst_cam_look_at
        self.dst_azimuth_range = dst_azimuth_range
        self.dst_forward_offset = dst_forward_offset
        self.dst_pos_side_offset = dst_pos_side_offset
        self.dst_look_side_offset = dst_look_side_offset
        self.trajectory = trajectory
        self.move_time = move_time
        self.modal_time = modal_time
        self.camera_control = camera_control
        self.motion_bucket_range = motion_bucket_range
        self.cond_aug = cond_aug
        self.mock_dset_size = mock_dset_size
        self.reverse_prob = reverse_prob
        self.data_gpu = data_gpu
        self.spread_radius = spread_radius
        self.render_width = render_width
        self.render_height = render_height

        if len(self.split_json) == 0:
            all_scene_dns = sorted(os.listdir(self.dset_root))
            all_scene_dps = [os.path.join(self.dset_root, scene_dn)
                             for scene_dn in all_scene_dns]
            all_scene_dps = [scene_dp for scene_dp in all_scene_dps
                             if os.path.isdir(scene_dp) and 'scene' in scene_dp]
            all_scene_dps = all_scene_dps[start_idx:end_idx]
            all_scene_dns = [os.path.basename(scene_dp) for scene_dp in all_scene_dps]

            self.num_scenes = end_idx - start_idx
            self.start_idx = start_idx
            self.end_idx = end_idx

            assert len(all_scene_dns) == self.num_scenes, \
                f'{len(all_scene_dns)} vs {self.num_scenes}'

        else:
            print(f'[yellow]split_json ({split_json}) was given, so ignoring start_idx '
                  f'({start_idx}) and end_idx ({end_idx}) and applying split ({split}) instead')
            split_scenes_map = common.load_json(self.split_json)
            all_scene_dns = split_scenes_map[split]

            self.num_scenes = len(all_scene_dns)
            self.start_idx = 0
            self.end_idx = self.num_scenes

            print(f'[yellow]Using {self.num_scenes} scenes, ranging from '
                  f'{all_scene_dns[0]} to {all_scene_dns[-1]}')

        self.all_scene_dns = all_scene_dns
        self.avail_ego_views = 3
        self.avail_magic_views = 16
        self.avail_frames = 50
        self.avail_fps = 10

        ontology_fp = (fr'{self.dset_root}/scene_000000/ontology/'
                       fr'fdc593e7de5d680e0e82af9ca090e98a63d13a7e.json')
        with open(ontology_fp, 'r') as f:
            self.ontology = json.load(f)

        # semantic categories are deterministic (i.e. specified by the dataset).
        rnd_gen = np.random.default_rng(4)
        semantic_id_rgb_dict = {x['id']: (x['color']['r'], x['color']['g'], x['color']['b'])
                                for x in self.ontology['items']}
        semantic_id_rgb_map = np.zeros((np.max(list(semantic_id_rgb_dict.keys())) + 1, 3))
        for (k, v) in semantic_id_rgb_dict.items():
            semantic_id_rgb_map[k] = np.array(v) / 255.0

        # Append metadata with custom info for later processing.
        self.ontology['semantic_id_rgb_map'] = torch.tensor(semantic_id_rgb_map)

        self.next_example = None
        self.total_counter = mp.Value('i', 0)
        self.max_retries = 100
        self.reproject_rgbd = False

    def set_next_example(self, *args):
        '''
        For evaluation purposes.
        '''
        # Typically, args = [scene_idx, scene_dn, frame_skip, frame_start, reverse].
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

        # NOTE: This dataset export has some issues, such as some scene folders being incomplete
        # (e.g. sometimes only ~45 files instead of 50 with random missing RGB frames etc.).
        for retry_idx in range(self.max_retries):
            try:
                if self.next_example is not None:
                    print(f'[cyan]Loading next_example: {self.next_example}')
                    scene_idx = int(self.next_example[0])
                    scene_dn = str(self.next_example[1])
                    frame_skip = int(self.next_example[2])
                    frame_start = int(self.next_example[3])
                    reverse = bool(self.next_example[4])

                    if scene_idx < 0:
                        # This specific value means that we are choosing an out-of-distribution
                        # example, but we still need access to ParallelDomain data to load camera
                        # parameters.
                        scene_dn = 'scene_000000'

                else:
                    if retry_idx >= 1 or self.force_shuffle:
                        # Try again with a random index.
                        idx2 = np.random.randint(0, self.mock_dset_size)
                        idx = (idx2 + idx) % self.mock_dset_size
                    scene_idx = idx % self.num_scenes + self.start_idx
                    scene_dn = self.all_scene_dns[scene_idx - self.start_idx]

                    # Choose a random speed for each clip subsampled from the video.
                    frame_skip = np.random.randint(1, 3)  # Hardcoded bounds; becomes 1 or 2.

                    # Sample a random offset to avoid always starting at the same frame.
                    cover_video = frame_skip * (Tcm - 1) + 1  # Number of frames; inclusive.
                    max_frame_start = Tv - cover_video - 1  # Highest possible offset; inclusive.
                    frame_start = np.random.randint(0, max_frame_start + 1)

                    # Apply random temporal data augmentations.
                    reverse = (np.random.rand() < self.reverse_prob)

                scene_dp = os.path.join(self.dset_root, scene_dn)
                pcl_dp = os.path.join(self.pcl_root, scene_dn)

                # Obtain resulting final clip frame indices.
                fps = int(round(self.avail_fps / frame_skip))  # Becomes 10 or 5.
                clip_frames = np.arange(Tcm) * frame_skip + frame_start
                if scene_idx >= 0:
                    assert 0 <= clip_frames[0] and clip_frames[-1] <= Tv - 1, str(clip_frames)

                if reverse:
                    clip_frames = clip_frames[::-1].copy()

                # Load scene metadata and rendered viewpoint camera matrices.
                calibration_fp = glob.glob(os.path.join(scene_dp, 'calibration', '*.json'))[0]
                calibration = common.load_json(calibration_fp)
                (view_names, all_intrinsics, all_extrinsics) = \
                    geometry.get_pardom_camera_matrices_torch(calibration)
                # (V, 3, 3) tensor of float32; (V, 4, 4) tensor of float32.
                # NOTE: First 16 views are magic (lower FOV), last 3 views are ego (higher FOV).

                # Load 3D point cloud video clip (requires dataset conversion beforehand!).
                if scene_idx >= 0:
                    pcl_dict = self.load_point_clouds(
                        pcl_dp, clip_frames, verbose)
                else:
                    pcl_dict = None

                # Define input and output camera trajectories.
                (extrinsics_src, extrinsics_dst, intrinsics_src, intrinsics_dst,
                 readable_angles, src_view_idx, dst_view_idx, motion_amount) = \
                    self.sample_trajectories(all_extrinsics, all_intrinsics, verbose)
                # (Tcm, 3); 2x (Tcm, 4, 4); 2x (Tcm, 3, 3) tensor of float32.

                # Load and/or synthesize input and viewpoint videos.
                # NOTE: We have to call different methods depending on whether the input/output
                # mode is a given view (available from dataset) or a pseudo-GT view.
                if scene_idx >= 0:
                    with torch.no_grad():
                        if self.input_mode in ['ego_forward', 'magic_random']:
                            rgb_src = self.load_src_rgb(
                                scene_dp, clip_frames, src_view_idx, verbose)
                            rgb_src = torch.tensor(rgb_src)
                            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

                        elif self.input_mode in ['traffic1']:
                            (rgb_src, _) = self.synth_rgb(
                                pcl_dict, self.input_modality,
                                extrinsics_src, intrinsics_src, calc_reproject=False)
                            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

                        if self.output_mode in ['magic_opposite']:
                            rgb_dst = self.load_dst_rgb(
                                scene_dp, clip_frames, dst_view_idx, verbose)
                            rgb_dst = torch.tensor(rgb_dst)
                            reproject = None
                            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

                        elif self.output_mode in ['topdown1', 'topdown2', 'traffic1']:
                            (rgb_dst, reproject) = self.synth_rgb(
                                pcl_dict, self.output_modality,
                                extrinsics_dst, intrinsics_dst, calc_reproject=True)
                            # 2x (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
                else:
                    rgb_src = None
                    rgb_dst = None
                    reproject = None

                # Now construct the final tensors that will be actually used in the model pipeline.
                data_dict = self.construct_dict(
                    rgb_src, rgb_dst, reproject, fps, readable_angles,
                    src_view_idx, dst_view_idx,
                    extrinsics_src, extrinsics_dst, intrinsics_src, intrinsics_dst,
                    motion_amount, verbose)

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
        data_dict['dset'] = torch.tensor([2])
        data_dict['idx'] = torch.tensor([idx])
        data_dict['scene_idx'] = torch.tensor([scene_idx])
        data_dict['frame_start'] = torch.tensor([frame_start])
        data_dict['frame_skip'] = torch.tensor([frame_skip])
        data_dict['clip_frames'] = torch.tensor(clip_frames)

        if verbose:
            print(f'[gray]ParallelDomainSynthViewDataset __getitem__ '
                  f'idx: {idx} scene_idx: {scene_idx} took: {time.time() - start_time:.3f} s')

        return data_dict

    def load_src_rgb(self, scene_dp, clip_frames, src_view_idx, verbose):
        if self.input_mode == 'ego_forward':
            rgb_src = common.load_pardom_video_vis_frames(
                scene_dp, self.input_modality, 'ego', 1, self.ontology,
                clip_frames, self.center_crop, self.frame_width, self.frame_height)
            # (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].

        elif self.input_mode == 'magic_random':
            rgb_src = common.load_pardom_video_vis_frames(
                scene_dp, self.input_modality, 'magic', src_view_idx, self.ontology,
                clip_frames, self.center_crop, self.frame_width, self.frame_height)
            # (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].

        return rgb_src

    def load_dst_rgb(self, scene_dp, clip_frames, dst_view_idx, verbose):
        assert self.move_time == 0, \
            f'Expected move_time == 0 but got {self.move_time}'

        if self.output_mode == 'magic_opposite':
            rgb_dst = common.load_pardom_video_vis_frames(
                scene_dp, self.output_modality, 'magic', dst_view_idx, self.ontology,
                clip_frames, self.center_crop, self.frame_width, self.frame_height)
            # (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].

        return rgb_dst

    def load_point_clouds(self, pcl_dp, clip_frames, verbose):
        all_xyz = []
        all_rgb = []
        all_segm = []
        all_tag = []  # This is actually view index, but renamed for disambiguation.

        for t in clip_frames:
            pcl_fp = os.path.join(pcl_dp, f'pcl_rgb_segm_{t * 10 + 5:06d}.pt')
            pcl_all = torch.load(pcl_fp)

            (pcl_xyz, pcl_rgb, pcl_segm, pcl_tag) = pcl_all
            # (V, N, 3) of float16; (V, N, 3) of uint8;
            # (V, N, 1) of uint8; (V, N, 1) of uint8.

            all_xyz.append(pcl_xyz)
            all_rgb.append(pcl_rgb)
            all_segm.append(pcl_segm)
            all_tag.append(pcl_tag)

        # Do not stack, but keep lists, for efficiency.
        pcl_dict = dict()
        pcl_dict['xyz'] = all_xyz  # List of (V, N, 3) tensor of float16.
        pcl_dict['rgb'] = all_rgb  # List of (V, N, 3) tensor of uint8.
        pcl_dict['segm'] = all_segm  # List of (V, N, 1) tensor of uint8.
        pcl_dict['tag'] = all_tag  # List of (V, N, 1) tensor of uint8.

        return pcl_dict

    def sample_trajectories(self, avail_extrinsics, avail_intrinsics, verbose):
        '''
        :param avail_extrinsics: (V, 4, 4) tensor of float32.
        :param avail_intrinsics: (V, 3, 3) tensor of float32.
        :return readable_angles: (Tcm, 3) tensor of float32.
        :return extrinsics_src: (Tcm, 4, 4) tensor of float32.
        :return extrinsics_dst: (Tcm, 4, 4) tensor of float32.
        :return intrinsics_src: (Tcm, 3, 3) tensor of float32.
        :return intrinsics_dst: (Tcm, 3, 3) tensor of float32.
        :return motion_amount: float32 in [0, 1].
        '''
        Tcm = self.model_frames

        assert self.input_mode in ['ego_forward', 'magic_random', 'traffic1'], \
            f'input_mode: {self.input_mode} vs ego_forward or magic_random or traffic1'
        assert self.output_mode in ['topdown1', 'topdown2', 'magic_opposite', 'traffic1'], \
            f'output_mode: {self.output_mode} vs topdown1 or topdown2 or magic_opposite or traffic1'

        if self.next_example is not None:
            # Nothing here is random yet; perhaps will do magic stuff later.
            pass

        # Determine source camera trajectory.
        # NOTE: We determine viewing directions by defining a pair of positions over time.
        # NOTE: Default ego forward is [1.60, 0.0, 1.55], and first magic is [-42.0, 0.0, 6.0].
        src_view_idx = -1

        if self.input_mode == 'ego_forward':
            position_src = np.array([1.60, 0.0, 1.55], dtype=np.float32)
            position_src = np.tile(position_src[None], (self.model_frames, 1))
            look_at_src = np.array([6.60, 0.0, 1.55], dtype=np.float32)
            look_at_src = np.tile(look_at_src[None], (self.model_frames, 1))
            # 2x (Tcm, 3) array of float32.

        elif self.input_mode == 'magic_random':
            src_view_idx = np.random.randint(0, self.avail_magic_views)
            # NOTE: The matrices are not important in this case, since we will load frames from disk.
            position_src = np.array([avail_extrinsics[src_view_idx, 0, 3],
                                     avail_extrinsics[src_view_idx, 1, 3],
                                     avail_extrinsics[src_view_idx, 2, 3]], dtype=np.float32)
            position_src = np.tile(position_src[None], (self.model_frames, 1))
            # NOTE: I am uncertain about with which values the PD dataset was generated exactly:
            look_at_src = np.array([0.0, 0.0, -2.0], dtype=np.float32)
            look_at_src = np.tile(look_at_src[None], (self.model_frames, 1))
            # 2x (Tcm, 3) array of float32.

        elif self.input_mode == 'traffic1':
            (position_src, look_at_src, azimuth_src_deg, height_src, radius_src) = \
                self.sample_traffic1(avail_extrinsics, avail_intrinsics)
            # 2x (Tcm, 3) array of float32.

        # Determine destination camera trajectory.
        dst_view_idx = -1
        readable_angles = np.zeros(3, dtype=np.float32)
        readable_angles = np.tile(readable_angles[None], (self.model_frames, 1))
        # (Tcm, 3) array of float32.

        if self.output_mode == 'topdown1':
            assert self.dst_azimuth_range == [0.0, 0.0], self.dst_azimuth_range

            # Default: [-8.0, 0.0, 8.0].
            position_dst = np.array(self.dst_cam_position, dtype=np.float32)
            position_dst = np.tile(position_dst[None], (self.model_frames, 1))
            # Default: [5.60, 0.0, 1.55].
            look_at_dst = np.array(self.dst_cam_look_at, dtype=np.float32)
            look_at_dst = np.tile(look_at_dst[None], (self.model_frames, 1))
            # 2x (Tcm, 3) array of float32.

        elif self.output_mode == 'topdown2':
            azimuth_deg = np.random.uniform(*self.dst_azimuth_range)
            azimuth_rad = np.deg2rad(azimuth_deg)

            # Default for azimuth = 0: position [-8.0, 0.0, 8.0] look at [5.60, 0.0, 1.55].
            # Desired for azimuth = 90: position [8.0, 9.0, 8.0] look at [8.0, -1.20, 1.55].
            # Desired for azimuth = -90: position [8.0, -9.0, 8.0] look at [8.0, 1.20, 1.55].

            unit_position = np.array([1.0 - np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0],
                                     dtype=np.float32)
            position_dst = np.array([unit_position[0]
                                     * (self.dst_forward_offset - self.dst_cam_position[0])
                                     + self.dst_cam_position[0],
                                     unit_position[1]
                                     * (self.dst_pos_side_offset - self.dst_cam_position[1])
                                     + self.dst_cam_position[1],
                                     self.dst_cam_position[2]], dtype=np.float32)
            look_at_dst = np.array([unit_position[0]
                                    * (self.dst_forward_offset - self.dst_cam_look_at[0])
                                    + self.dst_cam_look_at[0],
                                    unit_position[1]
                                    * (self.dst_look_side_offset - self.dst_cam_look_at[1])
                                    + self.dst_cam_look_at[1],
                                    self.dst_cam_look_at[2]], dtype=np.float32)
            position_dst = np.tile(position_dst[None], (self.model_frames, 1))
            look_at_dst = np.tile(look_at_dst[None], (self.model_frames, 1))
            # 2x (Tcm, 3) array of float32.

            readable_angles = np.array([azimuth_deg * np.pi / 180.0, 0.0, 0.0], dtype=np.float32)
            readable_angles = np.tile(readable_angles[None], (self.model_frames, 1))
            # (Tcm, 3) array of float32.

        elif self.output_mode == 'magic_opposite':
            assert self.input_mode == 'magic_random', \
                f'Expected input_mode == magic_random but got {self.input_mode}'
            dst_view_idx = (src_view_idx + (self.avail_magic_views // 2)) % self.avail_magic_views
            # NOTE: The matrices are not important in this case, since we will load frames from disk.
            position_dst = np.array([avail_extrinsics[dst_view_idx, 0, 3],
                                     avail_extrinsics[dst_view_idx, 1, 3],
                                     avail_extrinsics[dst_view_idx, 2, 3]], dtype=np.float32)
            position_dst = np.tile(position_dst[None], (self.model_frames, 1))
            # NOTE: I am uncertain about with which values the PD dataset was generated exactly:
            look_at_dst = np.array([0.0, 0.0, -2.0], dtype=np.float32)
            look_at_dst = np.tile(look_at_dst[None], (self.model_frames, 1))
            # 2x (Tcm, 3) array of float32.

            readable_angles = np.array([np.pi, 0.0, 0.0], dtype=np.float32)
            readable_angles = np.tile(readable_angles[None], (self.model_frames, 1))
            # (Tcm, 3) array of float32.

        elif self.output_mode == 'traffic1':
            assert self.input_mode == 'traffic1', \
                f'Expected input_mode == traffic1 but got {self.input_mode}'
            (position_dst, look_at_dst, azimuth_dst_deg, height_dst, radius_dst) = \
                self.sample_traffic1(avail_extrinsics, avail_intrinsics,
                                     azimuth_src_deg=azimuth_src_deg)
            # 2x (Tcm, 3) array of float32.

            readable_angles = np.array([(azimuth_dst_deg - azimuth_src_deg) * np.pi / 180.0,
                                        height_dst - height_src,
                                        radius_dst - radius_src],
                                       dtype=np.float32)
            readable_angles = np.tile(readable_angles[None], (self.model_frames, 1))
            # (Tcm, 3) array of float32.

        # Let's keep it this way for simplicity, since I barely noticed an effect in Kubric.
        motion_amount = 0.5

        # NOTE: Magic has slightly wider FOV than ego.
        # Let's use ego intrinsics for everything to mitigate tiny objects.
        intrinsics_src = avail_intrinsics[-2:-1].tile((Tcm, 1, 1))
        intrinsics_dst = avail_intrinsics[-2:-1].tile((Tcm, 1, 1))
        # 2x (Tcm, 3, 3) tensor of float32.

        # Determine actual target camera trajectory (i.e. by interpolating if configured).
        if self.move_time >= 1:
            for t in range(0, self.move_time):
                position_start = position_src[t].copy()
                position_end = position_dst[t].copy()
                look_at_start = look_at_src[t].copy()
                look_at_end = look_at_dst[t].copy()
                intrinsics_start = intrinsics_src[t].clone()
                intrinsics_end = intrinsics_dst[t].clone()

                if self.trajectory == 'interpol_linear':
                    alpha = t / self.move_time
                elif self.trajectory == 'interpol_sine':
                    alpha = (1.0 - np.cos(t / self.move_time * np.pi)) / 2.0
                else:
                    raise ValueError(f'Unknown trajectory: {self.trajectory}')

                if not (self.input_mode == 'traffic1' and self.output_mode == 'traffic1'):
                    position_dst[t] = position_start * (1.0 - alpha) + position_end * alpha
                else:
                    position_dst[t] = geometry.interpolate_spherical(
                        position_start, position_end, alpha)
                    if verbose and t == self.move_time // 2:
                        print(f'[gray]Interpolating camera position in spherical coordinate space '
                              f'because of traffic1 mode; position_start: {position_start}  '
                              f'position_end: {position_end}  alpha: {alpha:.2f}  '
                              f'position_dst: {position_dst[t]}')

                look_at_dst[t] = look_at_start * (1.0 - alpha) + look_at_end * alpha
                intrinsics_dst[t] = intrinsics_start * (1.0 - alpha) + intrinsics_end * alpha
        # 2x (Tcm, 3); (Tcm, 3, 3) array of float32.

        # Convert all to camera extrinsics over time.
        extrinsics_src = np.zeros((Tcm, 4, 4), dtype=np.float32)
        extrinsics_dst = np.zeros((Tcm, 4, 4), dtype=np.float32)
        for t in range(0, Tcm):
            extrinsics_src[t] = geometry.extrinsics_from_look_at(position_src[t], look_at_src[t])
            extrinsics_dst[t] = geometry.extrinsics_from_look_at(position_dst[t], look_at_dst[t])

        readable_angles = torch.tensor(readable_angles, dtype=torch.float32)
        extrinsics_src = torch.tensor(extrinsics_src, dtype=torch.float32)
        extrinsics_dst = torch.tensor(extrinsics_dst, dtype=torch.float32)

        # We normalize intrinsics to [0, 1]^2.
        intrinsics_src[:, 0, :] /= 640
        intrinsics_src[:, 1, :] /= 480
        intrinsics_dst[:, 0, :] /= 640
        intrinsics_dst[:, 1, :] /= 480

        return (extrinsics_src, extrinsics_dst, intrinsics_src, intrinsics_dst,
                readable_angles, src_view_idx, dst_view_idx, motion_amount)

    def sample_traffic1(self, avail_extrinsics, avail_intrinsics, azimuth_src_deg=None):
        # From a random angle, height, and radius, look down at ego vehicle.
        # NOTE: We enforce geometric locality between input and output views
        # only in terms in terms of azimuth, but not height or radius (unlike Kubric-4D).
        if azimuth_src_deg is None:
            azimuth_deg = np.random.uniform(0.0, 360.0)
        else:
            azimuth_deg = azimuth_src_deg + np.random.uniform(*self.dst_azimuth_range)
        azimuth_rad = np.deg2rad(azimuth_deg)
        height = np.random.uniform(4.0, 12.0)
        radius = np.random.uniform(8.0, 22.0)

        position = np.array([radius * np.cos(azimuth_rad),
                             radius * np.sin(azimuth_rad),
                             height], dtype=np.float32)
        position = np.tile(position[None], (self.model_frames, 1))
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        look_at = np.tile(look_at[None], (self.model_frames, 1))
        # 2x (Tcm, 3) array of float32.

        return (position, look_at, azimuth_deg, height, radius)

    def synth_rgb(self, pcl_dict, modality, extrinsics, intrinsics, calc_reproject=False):
        '''
        :param pcl_dict: dict.
        :param modality: 'rgb' or 'segm'.
        :param extrinsics: (Tcm, 4, 4) tensor of float32.
        :param intrinsics: (Tcm, 3, 3) tensor of float32.
        :return rgb: (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].
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
        pcl_segm = pcl_dict['segm']  # List of (V, N, 1) tensor of uint8.
        pcl_tag = pcl_dict['tag']  # List of (V, N, 1) tensor of uint8.

        # NOTE: We normalized the intrinsics previously to [0, 1]^2, so we have to convert it to
        # unnormalized pixel coordinates here.
        used_intrinsics = intrinsics.clone()
        used_intrinsics[:, 0, :] *= self.render_width
        used_intrinsics[:, 1, :] *= self.render_height

        # We also have to correct the aspect ratio, which corresponds to cropping (in the exact same
        # manner as the earlier frame loading pipeline) to avoid stretching!
        old_ar = 640.0 / 480.0
        new_ar = self.render_width / self.render_height
        if new_ar > old_ar + 1e-3:
            # Because of how camera projection works, both operations result in equivalent outcomes:
            # used_intrinsics[:, 1, 1] *= new_ar / old_ar
            used_intrinsics[:, 1, 1] = used_intrinsics[:, 0, 0]
        elif new_ar < old_ar - 1e-3:
            used_intrinsics[:, 0, 0] = used_intrinsics[:, 1, 1]

        rgb = []
        if calc_reproject and self.reproject_rgbd:
            reproject = []
        else:
            reproject = None

        for t in range(Tcm):
            cur_xyz = pcl_xyz[t].type(torch.float32)  # (V, N, 3) tensor of float32.
            cur_xyz = cur_xyz.to(device)
            # NOTE: Due to being stored in float16 format and some PD points being extremely far
            # away (the scenes are basically unbounded), some values are huge or infinity.
            # cur_xyz = torch.clamp(cur_xyz, -64.0, 64.0)
            cur_rgb = (pcl_rgb[t].to(device) / 255.0).type(torch.float32)

            if modality == 'rgb':
                cur_vis = cur_rgb
                # (V, N, 3) tensor of float32.

            elif modality == 'segm':
                # Gradually interpolate visual features from color to other tasks.
                if 0 < t or self.modal_time == 0:
                    semantic_id_rgb_map = self.ontology['semantic_id_rgb_map']
                    semantic_id_rgb_map = semantic_id_rgb_map.to(device)
                    # (256, 3) tensor of float32.

                    cur_segm = pcl_segm[t]  # (V, N, 1) tensor of uint8.
                    cur_segm = cur_segm.to(device)
                    cur_segm_flat = cur_segm.flatten().type(torch.long)
                    # (V * N) tensor of int64.

                    cur_segm_rgb_flat = semantic_id_rgb_map[cur_segm_flat]
                    cur_segm_rgb = cur_segm_rgb_flat.reshape(*cur_segm.shape[:2], 3)
                    # (V, N, 3) tensor of float32.

                if 0 < t < self.modal_time:
                    alpha = t / self.modal_time
                    cur_vis = (1.0 - alpha) * cur_rgb + alpha * cur_segm_rgb
                elif t == 0 and 0 < self.modal_time:
                    cur_vis = cur_rgb
                else:
                    cur_vis = cur_segm_rgb
                # (V, N, 3) tensor of float32.

            else:
                raise ValueError(f'Unknown selected modality: {modality}')

            cur_vis = cur_vis.to(device)
            cur_xyzvis = torch.cat([cur_xyz, cur_vis], axis=-1)  # (V, N, 6) tensor of float32.

            if reproject is not None:
                # NOTE: Hardcoded forward ego viewpoint index value for stored point cloud data.
                src_xyzvis = cur_xyzvis[16]  # (N, 6) tensor of float32.

            del cur_xyz
            del cur_vis

            cur_xyzvis = cur_xyzvis.reshape(-1, 6)  # (V * N, 6) tensor of float32.
            cur_xyzvis = cur_xyzvis.type(torch.float64)  # (V * N, 6) tensor of float64.

            # This occurs at rendering resolution, which is ideally inbetween dataset and model.
            (cur_synth1, cur_weights, cur_uv, cur_depth) = geometry.project_points_to_pixels(
                cur_xyzvis, used_intrinsics[t], extrinsics[t],
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
            rgb.append(blur_synth1)
            del cur_synth1

            if reproject is not None:
                src_xyzvis = src_xyzvis.type(torch.float64)  # (N, 6) tensor of float64.
                (cur_synth2, cur_weights, cur_uv, cur_depth) = geometry.project_points_to_pixels(
                    src_xyzvis, used_intrinsics[t], extrinsics[t],
                    self.render_height, self.render_width, spread_radius=self.spread_radius)
                blur_synth2 = geometry.blur_into_black(
                    cur_synth2, kernel_size=reproject_blur_radius,
                    sigma=reproject_blur_radius / 4.0)
                blur_synth2 = rearrange(blur_synth2, 'h w c -> c h w')
                blur_synth2 = torch.nn.functional.interpolate(
                    blur_synth2[None], (self.frame_height, self.frame_width), mode='bilinear',
                    align_corners=False)[0]
                # (3, H, W) tensor of float32 in [0, 1].

                blur_synth2 = blur_synth2.cpu()
                reproject.append(blur_synth2)
                del cur_synth2

            del cur_xyzvis
            del cur_weights
            del cur_uv
            del cur_depth

        rgb = torch.stack(rgb, dim=0)
        rgb = rgb * 2.0 - 1.0
        # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

        if reproject is not None:
            reproject = torch.stack(reproject, dim=0)
            reproject = reproject * 2.0 - 1.0
            # (Tcm, 3, Hp, Wp) tensor of float32 in [-1, 1].

        return (rgb, reproject)

    def construct_dict(
            self, rgb_src, rgb_dst, reproject, fps, readable_angles, src_view_idx, dst_view_idx,
            extrinsics_src, extrinsics_dst, intrinsics_src, intrinsics_dst,
            motion_amount, verbose):
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
        # (Tcm) = (14) tensor of int32 = all 5 / 10.
        image_only_indicator = torch.zeros((1, Tcm), dtype=torch.float32)
        # (1, Tcm) = (1, 14) tensor of float32 = all 0.

        # Intended for visualization only, not for SphericalEmbedder:
        scaled_rel_angles = readable_angles.clone()  # Already in radians and/or meters.
        # (Tcm, 3) tensor of float32.

        # May be used by CameraEmbedder:
        scaled_rel_pose = torch.zeros((Tcm, 3, 4), dtype=torch.float32)
        for t in range(Tcm):
            RT1 = extrinsics_src[t]
            RT2 = extrinsics_dst[t]
            delta_RT = torch.linalg.inv(RT1) @ RT2
            scaled_rel_pose[t] = delta_RT[0:3, 0:4]  # .flatten()
        # (Tcm, 12) tensor of float32.

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
            data_dict['src_view_idx'] = torch.tensor([src_view_idx], dtype=torch.int32)
            data_dict['dst_view_idx'] = torch.tensor([dst_view_idx], dtype=torch.int32)

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


class ParallelDomainSynthViewModule(pl.LightningDataModule):

    def __init__(
            self, dset_root, train_videos, val_videos, test_videos,
            batch_size, num_workers, shuffle=True, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = ParallelDomainSynthViewDataset(
            dset_root, 'train', 0, train_videos, **kwargs)
        self.val_dataset = ParallelDomainSynthViewDataset(
            dset_root, 'val', train_videos, train_videos + val_videos, **kwargs)

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
