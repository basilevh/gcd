'''
Created by Basile Van Hoorick for GCD, 2024.
'''

import os  # noqa
import sys  # noqa
sys.path.insert(0, os.getcwd())  # noqa

# Library imports.
import argparse
import copy
import cv2
import fire
import gradio as gr
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
import torch
import tqdm
import tqdm.rich
import warnings
from einops import rearrange
from functools import partial
from lovely_numpy import lo
from rich import print
from tqdm import TqdmExperimentalWarning

# Internal imports.
from scripts import eval_utils
from sgm.data import common

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

_TITLE = '[{model_name}] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis'

_DESCRIPTION = '''
This demo showcases [GCD](https://gcd.cs.columbia.edu/).
We finetune a large-scale video diffusion model to accomplish monocular *dynamic view synthesis*.
In other words, we unlock the ability to rotate the camera and thereby generate novel viewpoints of any scene from just a single video.

Note that the uploaded video will be trimmed to a clip of 14 frames, and the model will output one or several samples of 14 frames each.
Gradio might sometimes be unresponsive -- please feel free to click "Visualize Angles" first to verify whether it works.
All results are saved to your disk for reproducibility and debugging purposes.

The currently loaded checkpoint path is `{model_path}`, and performs the following task:

***{task_desc}***.
'''

example_fns = ['kubmv7_scn01990_v10.mp4', 'kubmv7_scn01992_v11.mp4', 'kubmv7_scn01993_v11.mp4',
               'kubmv7_scn01995_v1.mp4', 'kubmv7_scn01996_v2.mp4', 'kubmv7_scn01997_v8.mp4',
               'pd_scn00033.mp4', 'pd_scn00340.mp4', 'pd_scn00506.mp4', 'pd_scn00590.mp4',
               'pd_scn01061.mp4', 'pd_scn01891.mp4', 'pd_scn01969.mp4', 'pd_scn02071.mp4']
example_signals = ['Video'] * 14
example_frame_offsets = [0] * 6 + [0] * 8
example_frame_strides = [2] * 6 + [1] * 8
example_frame_rates = [12] * 6 + [10] * 8

os.environ['GRADIO_TEMP_DIR'] = '/tmp/gradio_gcd'


def inference(model_bundle, input_rgb, azimuth_deg, elevation_deg, radius_m,
              num_samples, num_frames, input_frames,
              frame_rate, motion_bucket, cond_aug, decoding_t, use_ema, autocast,
              min_scale, max_scale, num_steps, force_custom_mbid):
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]

    # Verify dimensions.
    (Tc, _, Hp, Wp) = input_rgb.shape
    eval_utils.warn_resolution_mismatch(train_config, Wp, Hp)
    assert Hp % 64 == 0 and Wp % 64 == 0, \
        f'Input resolution must be a multiple of 64, but got {Hp} x {Wp}'

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, device, num_steps, num_frames,
        max_scale, min_scale, autocast, decoding_t)

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            pred_samples = []

            batch = eval_utils.construct_batch(
                input_rgb, azimuth_deg, elevation_deg, radius_m,
                input_frames, frame_rate, motion_bucket, cond_aug,
                force_custom_mbid, model_bundle, device)
            print('[gray]batch:', batch)

            for sample_idx in range(num_samples):
                # Perform denoising loop.
                video_dict = model.sample_video(
                    batch, enter_ema=use_ema, limit_batch=False)

                output_rgb = video_dict['sampled_video'].detach().cpu().numpy()
                # (Tc, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].

                pred_samples.append(output_rgb)

    return (pred_samples, batch)


def main_run(model_bundle, cam_vis, action, output_path, input_frames,
             azimuth_deg=30.0, elevation_deg=15.0, radius_m=1.0,
             raw_image=None, raw_video=None, which_one='Video',
             frame_offset=0, frame_stride=2, frame_rate=12,
             center_crop=True, resolution='384 x 256',
             num_samples=2, motion_bucket=127, cond_aug=0.02, decoding_t=14,
             use_ema=False, autocast=True,
             min_scale=1.0, max_scale=1.5, num_steps=25, force_custom_mbid=False):

    azimuth_deg = float(azimuth_deg)
    elevation_deg = float(elevation_deg)
    radius_m = float(radius_m)

    if 'rand' in action:
        azimuth_deg = np.round(np.random.uniform(*model_bundle[-7]) * 10.0) / 10.0
        elevation_deg = np.round(np.random.uniform(*model_bundle[-6]) * 10.0) / 10.0
        radius_m = np.round(np.random.uniform(*model_bundle[-5]) * 10.0) / 10.0

    if 'vis' in action:
        cam_vis.azimuth_change(azimuth_deg)
        cam_vis.polar_change(-elevation_deg)
        cam_vis.radius_change(radius_m / 8.0)
        new_fig = cam_vis.update_figure()
        description = ('The viewpoints are visualized on the top right. '
                       'Click Run Generation to update the results on the bottom right.')
        return (description, new_fig)

    if raw_image is not None:
        print('raw_image:')
        print(type(raw_image))
        print(lo(raw_image))

    if raw_video is not None:
        print('raw_video:')
        print(type(raw_video))
        print(lo(raw_video))

    frame_width = int(resolution.split('x')[0].strip())
    frame_height = int(resolution.split('x')[1].split('(')[0].strip())
    num_frames = 14
    clip_frames = np.arange(num_frames) * frame_stride + frame_offset

    print(f'which_one: {which_one}')

    if 'image' in which_one.lower():
        input_rgb = common.process_image(
            raw_image, center_crop, frame_width, frame_height, True)
        input_rgb = input_rgb[None].repeat(len(clip_frames), axis=0)
        input_rgb = (input_rgb + 1.0) / 2.0
        # (Tc, 3, Hp, Wp) array of float32 in [0, 1].

    elif 'video' in which_one.lower():
        input_rgb = common.load_video_mp4(
            raw_video, clip_frames, center_crop, frame_width, frame_height, True)
        input_rgb = (input_rgb + 1.0) / 2.0
        # (Tc, 3, Hp, Wp) array of float32 in [0, 1].

    (pred_samples, last_batch) = inference(
        model_bundle, input_rgb, azimuth_deg, elevation_deg, radius_m,
        num_samples, num_frames, input_frames, frame_rate,
        motion_bucket, cond_aug, decoding_t, use_ema, autocast,
        min_scale, max_scale, num_steps, force_custom_mbid)

    input_rgb = rearrange(input_rgb, 'T C H W -> T H W C')

    # Update 3D camera perspective visualization.
    vis_frame = (input_rgb[0] * 255.0).astype(np.uint8)
    cam_vis.azimuth_change(azimuth_deg)
    cam_vis.polar_change(-elevation_deg)
    cam_vis.radius_change(radius_m / 8.0)
    cam_vis.encode_image(vis_frame)
    new_fig = cam_vis.update_figure()

    fn_prefix = time.strftime('%Y%m%d-%H%M%S')
    fn_idx = 1
    model_name = model_bundle[4]

    components = []

    for (s, pred_sample) in enumerate(pred_samples):

        output_rgb = rearrange(pred_sample, 'T C H W -> T H W C')
        ioside_rgb = np.concatenate([input_rgb, output_rgb], axis=2)
        # (T, H, W*2, 3) array of float32 in [0, 1].

        # Pause a tiny bit at the beginning and end for less jerky looping.
        ioside_rgb = [ioside_rgb[0]] + list(ioside_rgb) + [ioside_rgb[-1]] * 2
        ioside_rgb = np.stack(ioside_rgb, axis=0)
        ioside_rgb = np.clip(ioside_rgb, 0.0, 1.0)
        # (T, H, W*2, 3) array of float32 in [0, 1].

        # Get unique file names to store videos to.
        out_vid_fp = None
        while out_vid_fp is None or os.path.exists(out_vid_fp):
            in_vid_fp = os.path.join(output_path,
                                     f'{fn_prefix}{fn_idx:02d}-{model_name}-in.mp4')
            out_vid_fp = os.path.join(output_path,
                                      f'{fn_prefix}{fn_idx:02d}-{model_name}-out-{s + 1}.mp4')
            ioside_vid_fp = os.path.join(output_path,
                                         f'{fn_prefix}{fn_idx:02d}-{model_name}-ioside-{s + 1}.mp4')
            if s == 0:
                fn_idx += 1
            else:
                break

        vis_fps = int(6 + frame_rate) // 2  # same as test.py and infer.py.
        if not os.path.exists(in_vid_fp):
            eval_utils.save_video(in_vid_fp, input_rgb, fps=vis_fps, quality=9)
        eval_utils.save_video(out_vid_fp, output_rgb, fps=vis_fps, quality=9)
        eval_utils.save_video(ioside_vid_fp, ioside_rgb, fps=vis_fps, quality=9)

        cur_ioside_output = gr.Video(
            value=ioside_vid_fp,
            format='mp4',
            label=f'Processed input and output video side by side (Sample {s + 1})',
            visible=True)
        cur_gen_output = gr.Video(
            value=out_vid_fp,
            format='mp4',
            label=f'Generated video from new viewpoint (Sample {s + 1})',
            visible=True)
        components.append(cur_ioside_output)
        components.append(cur_gen_output)

    for _ in range(4 - num_samples):
        components.append(gr.Video(visible=False))
        components.append(gr.Video(visible=False))

    motion_bucket_id = last_batch['motion_bucket_id'][0].item()
    fps_id = last_batch['fps_id'][0].item()
    cond_aug = last_batch['cond_aug'][0].item()
    description = f'''Done! {num_samples} sample(s) are shown on the right.

Debug info:
motion_bucket_id = {motion_bucket_id} / fps_id = {fps_id} / cond_aug = {cond_aug}'''
    if 'scaled_relative_angles' in last_batch:
        scaled_rel_angles = last_batch['scaled_relative_angles'].detach().cpu().tolist()
        description += f'''

scaled_relative_angles = {scaled_rel_angles[0]} to {scaled_rel_angles[1]}'''
    if 'scaled_relative_pose' in last_batch:
        scaled_rel_pose = last_batch['scaled_relative_pose'].detach().cpu().tolist()
        description += f'''

scaled_relative_pose = {scaled_rel_pose[0]} to {scaled_rel_pose[1]}'''

    to_return = [description, new_fig, *components]
    if 'angles' in action:
        to_return = [azimuth_deg, elevation_deg, radius_m] + to_return

    return to_return


def run_demo(device='cuda',
             debug=False, port=7880, support_ema=False,
             config_path='configs/infer_kubric.yaml',
             model_path='../pretrained/kubric_gradual_max90.ckpt',
             output_path='../eval/gradio_output/default/',
             examples_path='../eval/gradio_examples/',
             task_desc='(Unknown)',
             input_frames=14, resolution='384 x 256'):
    # Placeholder to be used in gradio components and actually loaded later.
    model_bundle = [None, None, None, None, 'stub', None, None, None]

    if not (os.path.exists(model_path)) and '*' in model_path:
        given_model_path = model_path
        model_path = sorted(glob.glob(model_path))[-1]
        print(f'[orange3]Warning: Parsed {given_model_path} '
              f'to assumed latest checkpoint {model_path}')

    # Initialize model.
    model_bundle = eval_utils.load_model_bundle(
        device, config_path, model_path, support_ema, verbose=True)
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]
    [azimuth_range, elevation_range, radius_range] = model_bundle[5:8]

    arbitrary_camera = (azimuth_range[0] != azimuth_range[1] and
                        elevation_range[0] != elevation_range[1] and
                        radius_range[0] != radius_range[1])

    os.makedirs(output_path, exist_ok=True)

    # NOTE: This part actually supports arbitrary spatial resolutions (W x H),
    # but both dimensions should be multiples of 64.
    resolution_options = ['1024 x 576 (Vanilla SVD)', '768 x 512', '576 x 384', '512 x 384',
                          '384 x 256']
    if resolution == '1024 x 576':
        default_resolution = '1024 x 576 (Vanilla SVD)'
    else:
        default_resolution = resolution + ' (Finetuned)'
        found = False
        for i in range(1, len(resolution_options)):
            if resolution in resolution_options[i]:
                resolution_options[i] = default_resolution
                found = True
                break
        if not found:
            resolution_options.append(default_resolution)

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE.format(model_name=model_name))

    with demo:
        gr.Markdown('# ' + _TITLE.format(model_name=model_name))
        gr.Markdown(_DESCRIPTION.format(model_path=model_path, task_desc=task_desc))

        with gr.Row():
            with gr.Column(scale=9, variant='panel'):

                which_rad = gr.Radio(
                    ['Image (becomes static video)', 'Video'],
                    value='Video',
                    label='Select which input to use')
                image_block = gr.Image(
                    type='numpy', image_mode='RGB',
                    sources=['upload', 'webcam', 'clipboard'],
                    visible=True,
                    label='Raw input image')
                video_block = gr.Video(
                    sources=['upload', 'webcam'],
                    include_audio=False,
                    visible=True,
                    label='Raw input video')

                gr.Markdown('*Video clip processing options:*')
                frame_offset_sld = gr.Slider(
                    0, 100, value=0, step=1,
                    label='Frame offset (start later)')
                frame_stride_sld = gr.Slider(
                    1, 10, value=2, step=1,
                    label='Frame stride (temporally subsample)')
                frame_rate_sld = gr.Slider(
                    5, 30, value=12, step=1,
                    label='Frame rate (after subsampling)')
                resolution_rad = gr.Radio(
                    resolution_options,
                    value=default_resolution,
                    label='Select resolution to resize to and run model inference at')
                center_crop_chk = gr.Checkbox(
                    True, label='Center crop to correct aspect ratio')

                if arbitrary_camera:
                    gr.Markdown('*Try camera trajectory presets:*')
                    with gr.Row():
                        rotleft_btn = gr.Button('Rotate Left', variant='primary')
                        rotup_btn = gr.Button('Rotate Up', variant='primary')
                        rotright_btn = gr.Button('Rotate Right', variant='primary')
                    with gr.Row():
                        random_btn = gr.Button('Random Path', variant='primary')
                        zoomin_btn = gr.Button('Zoom In', variant='primary')
                        zoomout_btn = gr.Button('Zoom Out', variant='primary')

                # if arbitrary_camera:
                gr.Markdown('*Control camera trajectory manually:*')
                azimuth_sld = gr.Slider(
                    float(azimuth_range[0]), float(azimuth_range[1]),
                    value=(azimuth_range[0] + azimuth_range[1]) / 2.0,
                    step=5, label='Azimuth angle (horizontal rotation in degrees)')
                elevation_sld = gr.Slider(
                    float(elevation_range[0]), float(elevation_range[1]),
                    value=(elevation_range[0] + elevation_range[1]) / 2.0,
                    step=5, label='Elevation angle (vertical rotation in degrees)')
                radial_sld = gr.Slider(
                    float(radius_range[0]), float(radius_range[1]),
                    value=(radius_range[0] + radius_range[1]) / 2.0,
                    step=0.1, label='Radial distance (relative zoom / translation from center)')

                gr.Markdown('*Model inference options:*')
                samples_sld = gr.Slider(
                    1, 4, value=1, step=1,
                    label='Number of samples to generate')

                with gr.Accordion('Advanced options', open=False):
                    motion_sld = gr.Slider(
                        0, 255, value=127, step=1,
                        label='Motion bucket (amount of flow to expect)')
                    cond_aug_sld = gr.Slider(
                        0.0, 0.2, value=0.02, step=0.01,
                        label='Conditioning noise augmentation strength')
                    decoding_sld = gr.Slider(
                        1, 14, value=14, step=1,
                        label='Number of output frames to simultaneously decode')
                    use_ema_chk = gr.Checkbox(
                        False, label='Use EMA (exponential moving average) model weights')
                    autocast_chk = gr.Checkbox(
                        True, label='Autocast (16-bit floating point precision)')
                    min_scale_sld = gr.Slider(
                        0.0, 5.0, value=1.0, step=0.1,
                        label='Diffusion guidance minimum (starting) scale')
                    max_scale_sld = gr.Slider(
                        0.0, 5.0, value=1.5, step=0.1,
                        label='Diffusion guidance maximum (ending) scale')
                    steps_sld = gr.Slider(
                        5, 100, value=25, step=5,
                        label='Number of diffusion inference timesteps')
                    custom_mbid_chk = gr.Checkbox(
                        False, label='Apply custom motion bucket ID value even if '
                        'trained with camera rotation magnitude synchronization')

                with gr.Row():
                    vis_btn = gr.Button('Visualize Angles', variant='secondary')
                    run_btn = gr.Button('Run Generation', variant='primary')

                desc_output = gr.Markdown(
                    'The results will appear on the right.')

            with gr.Column(scale=11, variant='panel'):

                vis_output = gr.Plot(
                    label='Relationship between input (green) and output (blue) camera poses')

                gen1_output = gr.Video(
                    format='mp4',
                    label='Generated video from new viewpoint (Sample 1)',
                    visible=True)
                gen2_output = gr.Video(
                    format='mp4',
                    label='Generated video from new viewpoint (Sample 2)',
                    visible=False)
                gen3_output = gr.Video(
                    format='mp4',
                    label='Generated video from new viewpoint (Sample 3)',
                    visible=False)
                gen4_output = gr.Video(
                    format='mp4',
                    label='Generated video from new viewpoint (Sample 4)',
                    visible=False)

                ioside1_output = gr.Video(
                    format='mp4',
                    label='Processed input and output video side by side (Sample 1)',
                    visible=True)
                ioside2_output = gr.Video(
                    format='mp4',
                    label='Processed input and output video side by side (Sample 2)',
                    visible=False)
                ioside3_output = gr.Video(
                    format='mp4',
                    label='Processed input and output video side by side (Sample 3)',
                    visible=False)
                ioside4_output = gr.Video(
                    format='mp4',
                    label='Processed input and output video side by side (Sample 4)',
                    visible=False)

        cam_vis = eval_utils.CameraVisualizer(vis_output)

        my_inputs = [image_block, video_block, which_rad,
                     frame_offset_sld, frame_stride_sld, frame_rate_sld,
                     center_crop_chk, resolution_rad,
                     samples_sld, motion_sld, cond_aug_sld, decoding_sld,
                     use_ema_chk, autocast_chk,
                     min_scale_sld, max_scale_sld, steps_sld, custom_mbid_chk]
        my_outputs = [desc_output, vis_output,
                      ioside1_output, gen1_output, ioside2_output, gen2_output,
                      ioside3_output, gen3_output, ioside4_output, gen4_output]

        vis_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'vis', output_path, input_frames),
                      inputs=[azimuth_sld, elevation_sld, radial_sld],
                      outputs=[desc_output, vis_output])

        run_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'run', output_path, input_frames),
                      inputs=[azimuth_sld, elevation_sld, radial_sld] + my_inputs,
                      outputs=my_outputs)

        preset_inputs = my_inputs
        preset_outputs = [azimuth_sld, elevation_sld, radial_sld] + my_outputs
        if arbitrary_camera:
            rotleft_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'angles_run',
                                         output_path, input_frames, azimuth_range[0], 0.0, 0.0),
                              inputs=preset_inputs, outputs=preset_outputs)
            rotup_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'angles_run',
                                       output_path, input_frames, 0.0, elevation_range[1], 0.0),
                            inputs=preset_inputs, outputs=preset_outputs)
            rotright_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'angles_run',
                                          output_path, input_frames, azimuth_range[1], 0.0, 0.0),
                               inputs=preset_inputs, outputs=preset_outputs)
            random_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'rand_angles_run',
                                        output_path, input_frames, 0.0, 0.0, 0.0),
                             inputs=preset_inputs, outputs=preset_outputs)
            zoomin_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'angles_run',
                                        output_path, input_frames, 0.0, 0.0, radius_range[0]),
                             inputs=preset_inputs, outputs=preset_outputs)
            zoomout_btn.click(fn=partial(main_run, model_bundle, cam_vis, 'angles_run',
                                         output_path, input_frames, 0.0, 0.0, radius_range[1]),
                              inputs=preset_inputs, outputs=preset_outputs)

        gr.Markdown('Try out an example below! Note that you still have to select '
                    'the desired angles, and then click Run Generation.')

        example_fps = [os.path.join(examples_path, x) for x in example_fns]

        examples_full = [list(x) for x in zip(example_fps, example_signals, example_frame_offsets,
                                              example_frame_strides, example_frame_rates)]
        print('examples_full:', examples_full)

        gr.Examples(
            examples=examples_full,  # NOTE: elements must match inputs list!
            inputs=[video_block, which_rad, frame_offset_sld, frame_stride_sld, frame_rate_sld],
            cache_examples=False,
            run_on_click=False,
            examples_per_page=50,
        )

    demo.queue(max_size=20)
    demo.launch(share=True, debug=debug, server_port=port)


if __name__ == '__main__':

    fire.Fire(run_demo)

    pass
