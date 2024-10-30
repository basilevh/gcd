# Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis

Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, Carl Vondrick

Columbia University, Stanford University, Toyota Research Institute

Published in ECCV 2024 (Oral)

[Paper](https://gcd.cs.columbia.edu/GCD_v4.pdf) | [Website](https://gcd.cs.columbia.edu/) | [Results](https://gcd.cs.columbia.edu/#results) | [Datasets](https://gcd.cs.columbia.edu/#datasets) | [Models](https://github.com/basilevh/gcd#pretrained-models)

https://github.com/user-attachments/assets/4be8ff92-a09a-442d-afb1-807226c1b2f9

This repository contains the Python code published as part of our paper _"[Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](https://gcd.cs.columbia.edu/GCD_v4.pdf)"_ (abbreviated **GCD**).

We provide setup instructions, pretrained models, inference code, training code, evaluation code, and dataset generation.

Please note that I refactored and cleaned the codebase for public release, mostly to simplify the structure as well as enhance readability and modularity, but I have not thoroughly vetted everything yet, so if you encounter any problems, please let us know by opening an issue, and feel free to suggest possible bugfixes if you have any.

Table of contents:

- [Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](#generative-camera-dolly-extreme-monocular-dynamic-novel-view-synthesis)
  - [Setup](#setup)
  - [Pretrained Models](#pretrained-models)
  - [Inference (Gradio)](#inference-gradio)
  - [Dataset Processing](#dataset-processing)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Custom Controls](#custom-controls)
    - [Metrics](#metrics)
    - [Custom Data (No GT)](#custom-data-no-gt)
  - [Dataset Generation](#dataset-generation)
    - [Kubric-4D](#kubric-4d)
    - [ParallelDomain-4D](#paralleldomain-4d)
    - [Data Visualization](#data-visualization)
  - [Citations](#citations)

## Setup

I recommend setting up a virtual environment and installing the necessary packages as follows:
```
conda create -n gcd python=3.10
conda activate gcd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/OpenAI/CLIP.git
pip install git+https://github.com/Stability-AI/datapipelines.git
pip install -r requirements.txt
```

The project was mostly developed with PyTorch version 2.0.1, however, it should work with later versions as well. In particular, I have experienced no issues so far with PyTorch 2.3.1, which is the latest version at the time of writing.

Note that the requirements file does _not_ specify package versions, since I am a fan of flexibility (adding version constraints would otherwise make adopting existing codebases in your new projects more cumbersome). If you experience any problems however, please let us know by opening an issue. I also provided the exact versions in `requirements_versions.txt` for your reference.

The subfolder `gcd-model` is originally based on the official [Stability AI generative models](https://github.com/Stability-AI/generative-models) repository.

## Pretrained Models

Below are the main **Kubric-4D** checkpoints we trained and used in our experiments, along with PSNR values on the test set. The left column indicates the maximum displacement of the camera in terms of horizontal rotation.

| Azimuth     | Gradual                                                                               | Direct                                                                               |
| ----------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Max 90 deg  | [Link](https://gcd.cs.columbia.edu/checkpoints/kubric_gradual_max90.ckpt) (17.88 dB)  | [Link](https://gcd.cs.columbia.edu/checkpoints/kubric_direct_max90.ckpt) (17.23 dB)  |
| Max 180 deg | [Link](https://gcd.cs.columbia.edu/checkpoints/kubric_gradual_max180.ckpt) (17.81 dB) | [Link](https://gcd.cs.columbia.edu/checkpoints/kubric_direct_max180.ckpt) (16.65 dB) |

Below are the main **ParallelDomain-4D** checkpoints that we trained and used in our experiments, along with PSNR or mIoU values on the test set. The left column indicates the predicted output modality (the input is always RGB).

| Modality    | Gradual                                                                              | Direct                                                                              |
| ----------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Color (RGB) | [Link](https://gcd.cs.columbia.edu/checkpoints/pardom_gradual_rgb.ckpt) (23.47 dB)   | [Link](https://gcd.cs.columbia.edu/checkpoints/pardom_direct_rgb.ckpt) (23.32 dB)   |
| Semantic    | [Link](https://gcd.cs.columbia.edu/checkpoints/pardom_gradual_semantic.ckpt) (39.0%) | [Link](https://gcd.cs.columbia.edu/checkpoints/pardom_direct_semantic.ckpt) (36.7%) |

All above checkpoints are 20.3 GB in size. Place them in `pretrained/` such that they have the same name as the corresponding config `.yaml` files.

## Inference (Gradio)

This section is for casually running our model on custom videos. For thorough quantitative evaluation on Kubric-4D or ParallelDomain-4D, or any command line inference outside of those two datasets that saves results and visualizations to your disk, please see the *Evaluation* section below instead.

For a **Kubric-4D** model, run:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0 python scripts/gradio_app.py --port=7880 \
--config_path=configs/infer_kubric.yaml \
--model_path=../pretrained/kubric_gradual_max90.ckpt \
--output_path=../eval/gradio_output/default/ \
--examples_path=../eval/gradio_examples/ \
--task_desc='Arbitrary monocular dynamic view synthesis on Kubric scenes up to 90 degrees azimuth'
```

To try out other models, simply change `config_path`, `model_path` and `task_desc`, for example for a **ParallelDomain-4D** model:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=1 python scripts/gradio_app.py --port=7881 \
--config_path=configs/infer_pardom.yaml \
--model_path=../pretrained/pardom_gradual_rgb.ckpt \
--output_path=../eval/gradio_output/default/ \
--examples_path=../eval/gradio_examples/ \
--task_desc='Upward monocular dynamic view synthesis on ParallelDomain scenes (RGB output)'
```

## Dataset Processing

For training and evaluation on either Kubric-4D and/or ParallelDomain-4D, you need to preprocess the datasets and store merged point clouds. This is because the datasets themselves only provide RGB-D videos from certain viewpoints, but we wish to fly around freely in the 4D scene and allow for learning arbitrary camera controls (and interpolating trajectories) as well.

For **Kubric-4D**:
```
cd data-gen/
python convert_pcl_kubric.py --gpus=0,0,1,1 --start_idx=0 --end_idx=3000 \
--input_root=/path/to/Kubric-4D/data \
--output_root=/path/to/Kubric-4D/pcl
```
Here, `/path/to/Kubric-4D/data` should be the folder that contains `scn00000`, `scn00001`, and so on. The script will read from `data` and write to `pcl/` (make sure you have 7.0 TB of free space).

For **ParallelDomain-4D**:
```
cd data-gen/
python convert_pcl_pardom.py --gpus=0,0,1,1 --start_idx=0 --end_idx=1600 \
--input_root=/path/to/ParallelDomain-4D/data \
--output_root=/path/to/ParallelDomain-4D/pcl
```
Here, `/path/to/ParallelDomain-4D/data` should be the folder that contains `scene_000000`, `scene_000001`, and so on. The script will read from `data/` and write to `pcl/` (make sure you have 4.4 TB of free space).

Both conversion scripts above mainly rely on GPUs for fast processing and can apply parallelization at the process level. For example, `--gpus=0,0,1,1` means spawn 4 workers (2 per GPU). During training, most of the disk I/O will be concentrated within the `pcl/` folder, so I recommend storing it on a fast, local SSD.

## Training

If you are training on your own dataset, I recommend creating a new data loader using the provided code as a reference. If you are using our data, please follow the *Dataset Processing* section above first.

First, download one of the two following available Stable Video Diffusion checkpoints: [SVD (14 frames)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd.safetensors) or [SVD-XT (25 frames)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt.safetensors), and place it in `pretrained/` (or update the checkpoint path in the config files referenced below). We work exclusively with the 14-frame version of SVD in our experiments due to resource constraints, so please change the other relevant config values if you are working with the 25-frame SVD-XT.

To start a GCD training run on **Kubric-4D** (gradual, max 90 deg):
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --base=configs/train_kubric_max90.yaml \
    --name=kb_v1 --seed=1234 --num_nodes=1 --wandb=0 \
    model.base_learning_rate=2e-5 \
    model.params.optimizer_config.params.foreach=False \
    data.params.dset_root=/path/to/Kubric-4D/data \
    data.params.pcl_root=/path/to/Kubric-4D/pcl \
    data.params.frame_width=384 \
    data.params.frame_height=256 \
    data.params.trajectory=interpol_linear \
    data.params.move_time=13 \
    data.params.camera_control=spherical \
    data.params.batch_size=4 \
    data.params.num_workers=4 \
    data.params.data_gpu=0 \
    lightning.callbacks.image_logger.params.batch_frequency=50 \
    lightning.trainer.devices="1,2,3,4,5,6,7"
```
To switch to a direct view synthesis model (without interpolation), adjust this value: `data.params.move_time=0`. To increase the maximum horizontal rotation (azimuth) angle, select the other config file: `train_kubric_max180.yaml`.

The resulting model will be able to perform 3-DoF monocular dynamic novel view synthesis on any RGB video, but will typically perform best within the Kubric domain and other videos that do not contain humans.

To start a GCD training run on **ParallelDomain-4D** (gradual, RGB):
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --base=configs/train_pardom_rgb.yaml \
    --name=pd_v1 --seed=1234 --num_nodes=1 --wandb=0 \
    model.base_learning_rate=2e-5 \
    model.params.optimizer_config.params.foreach=False \
    data.params.dset_root=/path/to/ParallelDomain-4D/data \
    data.params.pcl_root=/path/to/ParallelDomain-4D/pcl \
    data.params.split_json=../eval/list/pardom_datasplit.json \
    data.params.frame_width=384 \
    data.params.frame_height=256 \
    data.params.output_modality=rgb \
    data.params.trajectory=interpol_sine \
    data.params.move_time=13 \
    data.params.modal_time=0 \
    data.params.camera_control=none \
    data.params.batch_size=4 \
    data.params.num_workers=4 \
    data.params.data_gpu=0 \
    lightning.callbacks.image_logger.params.batch_frequency=50 \
    lightning.trainer.devices="1,2,3,4,5,6,7"
```
To switch to a direct view synthesis model (without interpolation), adjust this value: `data.params.move_time=0`. To change the output modality to semantic categories, select the other config file: `train_pardom_semantic.yaml`.

The resulting model will be able to perform upward monocular dynamic novel view synthesis on any RGB video, but will typically perform best on driving scenes (both synthetic and real) recorded facing forward at the street level. I also have internal models lying around that are capable of 3-DoF camera controls (ego-to-surround as well as surround-to-surround transforms) on this dataset, and although they are not part of experiments in our paper, I might showcase and/or release them here in the future.

Note that in all above commands, GPU index 0 (`data.params.data_gpu`) is reserved for constructing pairs of (input, ground truth) videos on the fly during data loading from the cached merged point clouds in `pcl/`. I recommend not training the network on the same GPU, which is why `lightning.trainer.devices` is disjoint and covers all remaining GPUs instead.

The VRAM usage for those will be around 50 GB per GPU in the provided examples. The three largest determining factors for VRAM are: (1) the batch size, (2) the spatial resolution (`frame_width` and `frame_height`), (3) the number of frames (SVD versus SVD-XT), and (4) whether EMA weight averaging is active. Most of our experiments were done on single nodes with 8x NVIDIA A100 or 8x NVIDIA A6000 devices, all without EMA due to limited compute.

Logs and visualizations will be stored to a dated subfolder within the `logs/` folder, which resides at the same level as `gcd-model/`. For each run, training visualizations are stored in the `visuals/` subfolder. If training ever gets interrupted, you can resume by pointing `--resume_from_checkpoint` to the latest valid checkpoint file, for example `--resume_from_checkpoint=../logs/2024-02-30T12-15-05_kb_v1/checkpoints/last.ckpt`.

## Evaluation

The following script generates many types of outputs for visual inspection and evaluation, and has to be adapted for each benchmark. For lighter operations, see the *Inference* section above. If you are using our data, make sure you followed the *Dataset Processing* section above first. If you are evaluating on your own custom dataset with ground truth, I recommend creating a new data loader and modifying the test script below.

To evaluate a GCD finetuned model on **Kubric-4D**, update the paths in `kubric_test20.txt` and run:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0,1 python scripts/test.py --gpus=0,1 \
--config_path=configs/infer_kubric.yaml \
--model_path=../logs/*_kb_v1/checkpoints/epoch=00000-step=00010000.ckpt \
--input=../eval/list/kubric_test20.txt \
--output=../eval/output/kubric_mytest1 \
--control_json=../eval/list/kubric_valtest_controls_gradual.json \
--control_idx=0 --autocast=1 --num_samples=2 --num_steps=25
```

For consistency and fairness, this command applies a deterministic set of camera angles and frame bounds associated with each scene, described in `kubric_valtest_controls_gradual.json`. These numbers were generated randomly only once and subsequently held fixed, but such that the input perspective (i.e. `spherical_src`) is aligned with view index 4 in the dataset. Change this to `kubric_valtest_controls_direct.json` if you are evaluating a direct view synthesis model. Also, you can evaluate over multiple samples both by increasing `--num_samples` (same controls) or by varying `--control_idx` (different controls per scene).

To evaluate a GCD finetuned model on **ParallelDomain-4D**, update the paths in `pardom_test20.txt` and run:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0,1 python scripts/test.py --gpus=0,1 \
--config_path=configs/infer_pardom.yaml \
--model_path=../logs/*_pd_v1/checkpoints/epoch=00000-step=00010000.ckpt \
--input=../eval/list/pardom_test20.txt \
--output=../eval/output/pardom_mytest1 \
--control_json=../eval/list/pardom_valtest_controls.json \
--control_idx=0 --autocast=1 --num_samples=2 --num_steps=25
```

Similarly as before, again for consistency and fairness, the control signals `pardom_valtest_controls.json` only contain frame bounds (i.e. offset and interval) for each scene.

In all cases, for the `--model_path` argument, `grep` is applied to deal with wildcards such that you do not need to worry about having to write dates. Corresponding ground truth frames are also rendered and stored in the output folder, allowing for numerical evaluations (see *Metrics* below).

### Custom Controls

If you want to ignore the provided JSON controls and instead run evaluation in a more freeform manner with chosen angles and frame bounds in Kubric-4D:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0,1 python scripts/test.py --gpus=0,1 \
--config_path=configs/infer_kubric.yaml \
--model_path=../logs/*_kb_v1/checkpoints/epoch=00000-step=00010000.ckpt \
--input=../eval/list/kubric_test20.txt \
--output=../eval/output/kubric_mytest2_cc \
--azimuth_start=70.0 --elevation_start=10.0 --radius_start=15.0 \
--delta_azimuth=30.0 --delta_elevation=15.0 --delta_radius=1.0 \
--frame_start=0 --frame_stride=2 --frame_rate=12 \
--reproject_rgbd=0 --autocast=1 --num_samples=2 --num_steps=25
```

In ParallelDomain-4D, the six pose-related arguments are not applicable, but video clip frame bounds can still be chosen.

### Metrics

The above `test.py` script saves per-scene `*_metrics.json` files under the `extra/` subfolder that contain overall as well as per-frame PSNR and SSIM numbers. It also saves all individual input, predicted, and target frames as images for each example processed by the model. Feel free to use these various outputs in your own quantitative evaluation workflow if you want to compute additional and/or aggregate metrics.

### Custom Data (No GT)

Compared to the main *Evaluation* section, this script does not depend on ground truth, which may not exist. Compared to the *Inference (Gradio)* section, this script exports more information and visualizations.

Prepare a direct path to either a video file or an image folder, or a list of either video files or image folders (in a `.txt` file with full paths), and run:
```
cd gcd-model/
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --gpus=0 \
--config_path=configs/infer_kubric.yaml \
--model_path=../pretrained/kubric_gradual_max90.ckpt \
--input=/path/to/video.mp4 \
--output=../eval/output/kubric_myinfer1 \
--delta_azimuth=30.0 --delta_elevation=15.0 --delta_radius=1.0 \
--frame_start=0 --frame_stride=2 --frame_rate=12 \
--autocast=1 --num_samples=2 --num_steps=25
```

Note that `--frame_rate` should reflect the target FPS *after* temporal subsampling of the input video, not *before*. If you want to evaluate multiple examples, I recommend using a list by setting `--input=/path/to/list.txt` to reduce the model loading overhead.

## Dataset Generation

If you want to use the same exact data as in our experiments, please see this [download link](https://gcd.cs.columbia.edu/#datasets) for a description and copies of Kubric-4D and ParallelDomain-4D. The rest of this section focuses on if you wish to tweak our pipeline and/or generate your own synthetic data.

### Kubric-4D

Follow [these instructions](https://openexr.com/en/latest/install.html) to install the OpenEXR library. Then, run the following commands to prepare your environment:
```
conda activate gcd
pip install bpy==3.4.0
pip install pybullet
pip install OpenEXR
cd data-gen/kubric/
pip install -e .
```

The subfolder `data-gen/kubric` is largely the same as [this commit](https://github.com/google-research/kubric/commit/e140e24e078d5e641c4ac10bf25743059bd059ce) from the official Google Research Kubric repository, but I added a minor bugfix to avoid race conditions when handling depth maps.

This is the command we used to generate the final Kubric-4D dataset (note the `rm -rf /tmp/` line):
```
cd data-gen/
for i in {1..110}
do
python export_kub_mv.py --mass_est_fp=gpt_mass_v4.txt \
--root_dp=/path/to/kubric_mv_gen \
--num_scenes=3000 --num_workers=10 --restart_count=30 \
--seed=900000 --num_views=16 --frame_width=576 --frame_height=384 \
--num_frames=60 --frame_rate=24 --save_depth=1 --save_coords=1 \
--render_samples_per_pixel=16 --focal_length=32 \
--fixed_alter_poses=1 --few_views=4
rm -rf /tmp/
done
```

The dataset is basically a variation of [TCOW Kubric](https://tcow.cs.columbia.edu/#datasets) and includes improvements such as more dynamic objects and increased mass realism. See the [TCOW supplementary](https://tcow.cs.columbia.edu/TCOW_v3.pdf) for details.

For the purposes of GCD, we render 16 synchronized multi-view videos from static cameras. Four viewpoints are at a high elevation of 45 degrees, and the other twelve viewpoints are at a low elevation of 5 degrees. I recommend inspecting `export_kub_mv.py` to gain more insight into its parameters and logic.

All scenes are generated i.i.d., so in our version of this dataset, we define the first 2800 as the training set and the last 100 + 100 as validation + test set respectively. The outer for loop regularly clears the `/tmp/` folder to avoid disk space issues.

### ParallelDomain-4D

This dataset comes from a [service](https://paralleldomain.com/) and cannot be regenerated. Please see the [download link](https://gcd.cs.columbia.edu/#datasets) for our copy.

Note that some scene folders do not exist (there are 1531 scene folders but the index goes up to 2143), and some scenes have a couple missing frames, which is why our dataloader is designed to be robust to both issues. You might see a few warning messages during training but this is normal. Also, unlike Kubric, the scenes are not decorrelated with respect to the index, hence in `pardom_datasplit.json` we pre-selected random subsets for training, validation, and testing.

We define the validation + test set sizes to be 61 + 61 scenes respectively (each roughly 4% of the total dataset).

### Data Visualization

I have written some tools, based on [TRI camviz](https://github.com/TRI-ML/camviz), to interactively visualize example scenes from Kubric-4D and ParallelDomain-4D on your local computer. I might release them here later, but feel free to contact me (Basile) in the meantime for the source code.

## Citations

If you use this codebase in your work (or any significant part of it, such as the changes needed to finetune SVD), please cite our paper:
```
@article{vanhoorick2024gcd,
    title={Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis},
    author={Van Hoorick, Basile and Wu, Rundi and Ozguroglu, Ege and Sargent, Kyle and Liu, Ruoshi and Tokmakov, Pavel and Dave, Achal and Zheng, Changxi and Vondrick, Carl},
    journal={European Conference on Computer Vision (ECCV)},
    year={2024}
}
```

I recommend also citing the original SVD paper:
```
@article{blattmann2023stable,
  title={Stable video diffusion: Scaling latent video diffusion models to large datasets},
  author={Blattmann, Andreas and Dockhorn, Tim and Kulal, Sumith and Mendelevitch, Daniel and Kilian, Maciej and Lorenz, Dominik and Levi, Yam and English, Zion and Voleti, Vikram and Letts, Adam and others},
  journal={arXiv preprint arXiv:2311.15127},
  year={2023}
}
```

If you use one of our datasets in your work, please also cite the respective source:
```
@article{greff2021kubric,
    title = {Kubric: a scalable dataset generator}, 
    author = {Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and Florian Golemo and Charles Herrmann and others},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022},
}
```

```
@misc{parallel_domain,
    title = {Parallel Domain},
    year = {2024},
    howpublished={\url{https://paralleldomain.com/}}
}
```
