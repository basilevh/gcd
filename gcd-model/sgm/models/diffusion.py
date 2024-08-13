
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
from peft.tuners.lora import layer as lora_layer
from ..modules.autoencoding.lpips.loss.lpips import LPIPS

import lovely_tensors
import numpy as np
from einops import rearrange
from lovely_numpy import lo
from rich import print
from skimage import metrics

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        ckpt_has_ema: bool = False,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        ablate_unet_scratch: bool = False,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        disable_loss_fn_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        ft_strategy: str = "everything",
    ):
        super().__init__()
        self.input_key = input_key
        self.log_keys = log_keys
        self.ablate_unet_scratch = ablate_unet_scratch

        # NOTE: This is typically overwritten in the SVD train config to become Adam.
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )

        # propagate these settings because it can't be overriden at command line
        # conditioner_config.params.emb_models[3].\
        #     params.disable_encoder_autocast = disable_first_stage_autocast
        # conditioner_config.params.emb_models[3]\
        #     .params.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        # print('conditioner_config.params.emb_models[3].params.disable_encoder_autocast:',
        #       conditioner_config.params.emb_models[3].params.disable_encoder_autocast)
        # print('conditioner_config.params.emb_models[3].params.en_and_decode_n_samples_a_time:',
        #       conditioner_config.params.emb_models[3].params.en_and_decode_n_samples_a_time)
        for (i, embedder) in enumerate(conditioner_config.params.emb_models):
            if hasattr(embedder.params, 'disable_encoder_autocast') and \
                    hasattr(embedder.params, 'en_and_decode_n_samples_a_time'):
                embedder.params.disable_encoder_autocast = disable_first_stage_autocast
                embedder.params.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
                print(f'conditioner_config.params.emb_models[{i}].params.disable_encoder_autocast:',
                      conditioner_config.params.emb_models[i].params.disable_encoder_autocast)
                print(f'conditioner_config.params.emb_models[{i}].params.en_and_decode_n_samples_a_time:',
                      conditioner_config.params.emb_models[i].params.en_and_decode_n_samples_a_time)

        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.disable_loss_fn_autocast = disable_loss_fn_autocast

        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        # NOTE: ckpt loading should be BEFORE EMA setup IF EMA is not stored inside.
        if ckpt_path is not None and not (ckpt_has_ema):
            print('[orange3]Loading checkpoint early, assuming it has no EMA...')
            self.init_from_ckpt(ckpt_path)

        if ft_strategy == "time":
            for name, parameter in self.model.diffusion_model.named_parameters():
                assert parameter.requires_grad
                if "time" not in name:
                    parameter.requires_grad = False
                else:
                    pass
        elif ft_strategy == "time_lora":
            self.model.diffusion_model.requires_grad_(False)

            def walk_adaptable_layers():
                for (
                    parent_name,
                    parent_module,
                ) in self.model.diffusion_model.named_modules():
                    for name, module in parent_module.named_children():
                        if "time" in name or "time" in parent_name:
                            if isinstance(module, nn.Linear):
                                yield (
                                    parent_name,
                                    parent_module,
                                    name,
                                    module,
                                    lora_layer.Linear,
                                )

            for parent_name, parent_module, name, module, adapter in list(
                    walk_adaptable_layers()):
                setattr(parent_module, name, adapter(module, "default", 16))

        elif ft_strategy == "everything":
            pass
        elif ft_strategy == "dummy":
            # just tune one paramater for debugging
            param_name = "output_blocks.11.1.time_mixer.mix_factor"

            for name, parameter in self.model.diffusion_model.named_parameters():
                assert parameter.requires_grad
                if param_name not in name:
                    parameter.requires_grad = False
                else:
                    pass
        else:
            raise NotImplementedError

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.no_cond_log = no_cond_log

        # NOTE: original code put ckpt loading always here which is wrong.
        if ckpt_path is not None and ckpt_has_ema:
            print('[orange3]Loading checkpoint late, assuming it comes with EMA...')
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self.lpips = LPIPS().eval()

        self.validation_step_outputs = []

    def init_from_ckpt(self, path: str,) -> None:
        assert os.path.exists(path) and os.path.isfile(path)
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        if self.ablate_unet_scratch:
            # Load only (frozen) VAE weights along with all embedders for conditioning,
            # but leave (trainable) U-Net randomly initialized / from scratch.
            print('[orange3]=> THIS IS AN ABLATION STUDY! ablate_unet_scratch = True')
            sd2 = {k: v for k, v in sd.items() if not ('diffusion' in k.lower())}
            print(f'[orange3]Went from {len(sd.keys())} to {len(sd2.keys())} keys')
            sd = sd2

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing "
              f"and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"[orange3]Missing Keys: {len(missing)}")
            print(f'First 10: {missing[0:10]}')
        if len(unexpected) > 0:
            print(f"[orange3]Unexpected Keys: {len(unexpected)}")
            print(f'First 5: {unexpected[0:5]}')

        print()

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples: (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples: (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples: (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        # import pdb
        # pdb.set_trace()

        with torch.autocast("cuda", enabled=not self.disable_loss_fn_autocast):
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)

        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        # x = (28, 3, 256, 384) of float32 in [-1, 1].

        x = self.encode_first_stage(x)
        # x = (28, 4, 32, 48) of float32 in [~-10.1, ~8.8].

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)

        return (loss, loss_dict)

    def training_step(self, batch, batch_idx):
        # NOTE: this is called by lightning in fit().

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            float(self.global_step),  # to suppress warnings
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def validation_step(self, batch, batch_idx):
        # NOTE: this is called by lightning in fit().
        all_lpips = []
        all_psnrs = []
        all_ssims = []

        with self.ema_scope("Validation"):
            with torch.no_grad():
                video_dict = self.sample_video(
                    batch, enter_ema=False, limit_batch=1, force_uc=False, keep_intermediate=False)
                n_frames_a_time = 1
                n_frames = video_dict["gt_video"].shape[0]  # This is actually B*T!

                assert n_frames % n_frames_a_time == 0
                for block in range(n_frames // n_frames_a_time):
                    gt_frames = video_dict["gt_video"][
                        block * n_frames_a_time: (block + 1) * n_frames_a_time]
                    pred_frames = video_dict["sampled_video"][
                        block * n_frames_a_time: (block + 1) * n_frames_a_time]
                    # 2x (1, 3, Hp, Wp) tensor of float32 in [0, 1].

                    gt_frames_numpy = gt_frames.detach().cpu().numpy()
                    pred_frames_numpy = pred_frames.detach().cpu().numpy()
                    # 2x (1, 3, Hp, Wp) array of float32 in [0, 1].

                    lpips = self.lpips(gt_frames * 2.0 - 1.0, pred_frames * 2.0 - 1.0).item()
                    psnr = metrics.peak_signal_noise_ratio(
                        gt_frames_numpy, pred_frames_numpy, data_range=1.0)
                    assert n_frames_a_time == 1
                    ssim = metrics.structural_similarity(
                        gt_frames_numpy[0], pred_frames_numpy[0], data_range=1.0, channel_axis=0)

                    all_lpips.append(lpips)
                    all_psnrs.append(psnr)
                    all_ssims.append(ssim)

        all_lpips = np.stack(all_lpips)
        all_psnrs = np.stack(all_psnrs)
        all_ssims = np.stack(all_ssims)

        assert all_psnrs.shape == all_lpips.shape
        metrics_dict = {
            'lpips': np.mean(all_lpips),
            'psnr': np.mean(all_psnrs),
            'ssim': np.mean(all_ssims),
        }
        print(f'[gray]validation_step: batch_idx = {batch_idx}, metrics = {metrics_dict}')
        self.validation_step_outputs.append(metrics_dict)

    def on_validation_epoch_end(self):
        outs = self.validation_step_outputs

        metrics_dict = {
            k: np.mean([t for out in outs for t in out[k].ravel().tolist()])
            for k in outs[0]
        }
        metrics_dict["step"] = self.global_step
        print(f'[gray]on_validation_epoch_end: metrics = {metrics_dict}')
        self.log_dict(metrics_dict, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        torch.cuda.empty_cache()  # attempt to save VRAM?
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        else:
            if context is not None:
                print(f"{context}: EMA is disabled; using training weights as is")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
        torch.cuda.empty_cache()  # attempt to save VRAM?

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        print('optimizer:', opt)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        def denoiser(input, sigma, c): return self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        # print([
        #     (embedder.input_key, batch[embedder.input_key].shape)
        #     for embedder in self.conditioner.embedders
        # ])
        # import pdb
        # pdb.set_trace()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):

                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)

                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()])
                             for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)

                    else:
                        # probably video, but log_img and log_images
                        # already take care of this.
                        # print("Got unexpected condition dim. Can't log ", x.shape)
                        xc = None

                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

                if xc is not None:
                    log[embedder.input_key] = xc

        return log

    def sample_video(self, batch, enter_ema=False, limit_batch=False):
        if enter_ema:
            with self.ema_scope('Sampling'):
                return self.sample_video(
                    batch, enter_ema=False, limit_batch=limit_batch)

        if isinstance(limit_batch, int) and limit_batch >= 1:
            # Only take and process first K (usually = 1) examples in batch to reduce VRAM.
            (B, T) = (batch['idx'].shape[0], batch['num_video_frames'])
            batch_old = batch
            batch = dict()
            batch.update({k: v[0:T * limit_batch] for (k, v) in batch_old.items()
                          if torch.is_tensor(v) and v.shape[0] >= B * T})
            batch.update({k: v[0:limit_batch] for (k, v) in batch_old.items()
                          if torch.is_tensor(v) and v.shape[0] < B * T})
            batch.update({k: v for k, v in batch_old.items()
                          if not (torch.is_tensor(v))})

        (c, uc) = self.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch,
            force_uc_zero_embeddings=['cond_frames', 'cond_frames_without_noise'])

        additional_model_inputs = {}
        additional_model_inputs['num_video_frames'] = batch['num_video_frames']
        additional_model_inputs['image_only_indicator'] = \
            batch['image_only_indicator'].repeat_interleave(2, dim=0)

        def denoiser(input, sigma, c):
            return self.denoiser(self.model, input, sigma, c, **additional_model_inputs)

        (BT, Cp, Hp, Wp) = batch['cond_frames'].shape
        assert Cp == 3
        Cl = 4
        F = 8
        (Hl, Wl) = (Hp // F, Wp // F)
        latent_shape = (BT, Cl, Hl, Wl)
        latent_noise = torch.randn(latent_shape, device=batch['cond_frames'].device)
        # (Tcm, 4, Hl, Wl) = (14, 4, 72, 128) tensor of float32.

        samples_z = self.sampler(denoiser, latent_noise, cond=c, uc=uc).detach()
        # (Tcm, 4, Hl, Wl) = (14, 4, 72. 128) tensor of float32 in [~-17.6, ~11.7].

        samples_x = self.decode_first_stage(samples_z).detach()
        # (Tcm, 3, Hp, Wp) = (14, 3, 576, 1024) tensor of float32 in [~-1.1, ~0.8].

        sampled_video = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        # (14, 3, 576, 1024) tensor of float32 in [0, 1].
        if 'jpg' in batch:
            gt_video = torch.clamp((batch['jpg'] + 1.0) / 2.0, min=0.0, max=1.0)
        else:
            gt_video = None
        # Optional (14, 3, 576, 1024) tensor of float32 in [0, 1].

        # NOTE: This is the direct conditioning signal WITH noise (conditioning augmentation).
        # Could be either a still frame (= first one in video) OR an entire video sequence.
        cond_video = torch.clamp((batch['cond_frames'] + 1.0) / 2.0, min=0.0, max=1.0)
        # (14, 3, 576, 1024) tensor of float32 in [0, 1].

        # Save some additional metadata (pick tiny tensors only).
        extra_info = {k: v.detach().cpu() for k, v in batch.items()
                      if torch.is_tensor(v) and v.shape.numel() <= 256}
        extra_info.update({k: v for k, v in batch.items() if not (torch.is_tensor(v))})

        # Organize & return results.
        video_dict = {
            'cond_video': cond_video,
            'sampled_z': samples_z,
            'sampled_video': sampled_video,
            'extra': extra_info,
        }
        if gt_video is not None:
            video_dict['gt_video'] = gt_video

        return video_dict

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        # NOTE: This is called by log_img() in main.py.

        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        # NOTE: This code seems designed more for images, but is still useful in SVD,
        # because it runs the ground truth through the VAE and back.
        x = self.get_input(batch)
        N = min(x.shape[0], N)

        x = x.to(self.device)[:N]
        z = self.encode_first_stage(x)
        x_cycle = self.decode_first_stage(z)

        # Later added for completeness.
        x_cond = batch['cond_frames'].to(self.device)[:N]
        z_cond = self.encode_first_stage(x_cond)
        x_cycle_cond = self.decode_first_stage(z_cond)

        log['target'] = x
        log['target_recon'] = x_cycle
        log['mycond'] = x_cond
        log['mycond_recon'] = x_cycle_cond
        log.update(self.log_conditionings(batch, N))

        if isinstance(self.first_stage_model.decoder, VideoDecoder):
            # Handle sampling for video a bit differently.
            video_dict = self.sample_video(
                batch, enter_ema=True, limit_batch=1)

            video_dict['vertcat'] = \
                torch.cat([video_dict['cond_video'], video_dict['sampled_video'],
                           video_dict['gt_video']], dim=2)
            # (14, 3, 1728, 1024) tensor of float32 in [0, 1].

            log['video_dict'] = video_dict

        else:
            raise NotImplementedError()

        return log
