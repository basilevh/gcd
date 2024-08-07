from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser

import numpy as np
from einops import rearrange
from rich import print


person_rgb_list = [[220, 20, 180],  # Animal
                   [64, 64, 64],  # Bicyclist
                   [128, 128, 128],  # Motorcyclist
                   [192, 192, 192],  # OtherRider
                   [220, 20, 60],  # Pedestrian
                   ]

# No Train because of large size.
vehicle_rgb_list = [[0, 60, 100],  # Bus
                    [0, 0, 142],  # Car
                    [0, 0, 90],  # Caravan/RV
                    [32, 32, 32],  # ConstructionVehicle
                    [119, 11, 32],  # Bicycle
                    [0, 0, 230],  # Motorcycle
                    [128, 230, 128],  # OwnCar
                    [0, 0, 70],  # Truck
                    [0, 64, 64],  # WheeledSlow
                    ]


def noncentral_checksum_np(arr):
    scaled = arr * (np.arange(arr.size).reshape(arr.shape) / (arr.size / 1000.0))
    return np.sum((scaled - 0.23) ** 2)


def noncentral_checksum(tensor, how=None):
    if how is None:
        if str(type(tensor)) == "<class 'torch.Tensor'>":
            how = "pytorch"
        else:
            how = "tf"

    if how == "pytorch":
        def f(t): return t.cpu().numpy()
    elif how == "tf":
        def f(t): return np.array(t)
    else:
        raise
    return noncentral_checksum_np(f(tensor))


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        harmonize_sigmas: bool = True,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        pd_person_weight=1.0,
        pd_vehicle_weight=1.0,
        focus_top=1.0,
        focus_steps=-1,
    ):
        super().__init__()

        self.harmonize_sigmas = harmonize_sigmas
        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.pd_person_weight = pd_person_weight
        self.pd_vehicle_weight = pd_vehicle_weight
        self.focus_top = focus_top
        self.focus_steps = focus_steps

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        # input = (28, 4, 32, 48) of float32 in [~-10.1, ~8.8].
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        # Important fix for correct SVD finetuning:
        # noise levels should be consistent across video frames!
        if self.harmonize_sigmas:
            old_sigmas = sigmas
            r_sigmas = rearrange(
                old_sigmas, "(b t) ... -> b t ...",
                t=additional_model_inputs['num_video_frames'])
            sigmas = r_sigmas[..., 0:1].broadcast_to(r_sigmas.shape).reshape(old_sigmas.shape)

        # new sigmas = (28) of float32 with B sequences of T repeating values.

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)

        loss_value = self.get_loss(model_output, input, w, batch)

        return loss_value

    def get_loss(self, model_output, target, w, batch):
        # print("output: ", noncentral_checksum(model_output.detach()))
        # print("target: ", noncentral_checksum(target.detach()))

        # Old:
        # if self.loss_type == "l2":
        #     return torch.mean(
        #         (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
        #     )
        # elif self.loss_type == "l1":
        #     return torch.mean(
        #         (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
        #     )
        # elif self.loss_type == "lpips":
        #     loss = self.lpips(model_output, target).reshape(-1)
        #     return loss
        # else:
        #     raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        # New for GCD:
        cur_step = batch['global_step']
        verbose = ((cur_step <= 60 and cur_step % 20 == 0) or cur_step % 100 == 0)

        diff = model_output - target
        # (BT, Cl, Hl, Wl) = (28, 4, 32, 48) of float32 in [~-6.6, ~6.7].
        BT = target.shape[0]

        if self.loss_type == "l2":
            loss_raw = diff ** 2
        elif self.loss_type == "l1":
            loss_raw = diff.abs()
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        # Apply manual loss weighting for important ParallelDomain classes.
        if self.pd_person_weight > 1.0 or self.pd_vehicle_weight > 1.0:
            loss_bias = torch.zeros_like(loss_raw)
            my_list = []

            if self.pd_person_weight > 1.0:
                my_list += [(x, self.pd_person_weight) for x in person_rgb_list]
            if self.pd_vehicle_weight > 1.0:
                my_list += [(x, self.pd_vehicle_weight) for x in vehicle_rgb_list]

            gt_rgb = batch['jpg']
            (Hp, Wp) = gt_rgb.shape[2:4]
            (Hl, Wl) = target.shape[2:4]
            threshold = 0.02

            for (rgb_val, weight) in my_list:
                rgb_torch = torch.tensor(rgb_val, dtype=torch.float32,
                                         device=target.device) / 127.5 - 1.0
                rgb_torch = rgb_torch[None, :, None, None]

                mask_pixel = ((gt_rgb - rgb_torch).abs().mean(dim=1, keepdim=True) < threshold)
                mask_pixel = mask_pixel.detach().type(torch.float32)
                # (BT, 1, Hp, Wp) = (28, 1, 256, 384) of float32 in [0, 1].

                # While the matching is not 100% precise, we simply average over all pixels in each
                # 8x8 square to obtain the final per-embedding loss weight.
                mask_latent = torch.nn.functional.interpolate(mask_pixel, (Hl, Wl), mode='area')
                loss_bias += loss_raw * mask_latent * (weight - 1.0)

            if verbose:
                print(f'=> PD loss at {cur_step} | loss_bias: {loss_bias} | '
                      f'loss_raw: {loss_raw}')

            loss_bias_mean = loss_bias.view(BT, -1).mean(dim=1)

        else:
            loss_bias = 0.0
            loss_bias_mean = 0.0

        if self.focus_steps > 0:
            cur_progress = np.clip(cur_step / self.focus_steps, 0.0, 1.0)
        else:
            cur_progress = 0.0

        # Incorporate half of category-weighted loss values BEFORE applying focal loss.
        loss_all = loss_raw + loss_bias * 0.5

        loss_all_mean = loss_all.view(BT, -1).mean(dim=1)

        # Apply adaptive top fraction focal loss.
        cur_top = (1.0 - cur_progress) + self.focus_top * cur_progress
        if cur_top < 1.0:
            diff_flat = loss_all.view(BT, -1)
            keep = int(diff_flat.shape[1] * cur_top)
            diff_top = diff_flat.topk(keep, dim=1)[0]
            loss_top = diff_top.mean(dim=1)

            # We also apply a small amount of regular loss to avoid overfitting to certain regions,
            # or possible instability.
            loss_focal = loss_top * 0.9 + loss_all_mean * 0.1
            # (BT).

            if verbose:
                print(f'=> focal loss at {cur_step} | loss_top: {loss_top} | '
                      f'loss_mean: {loss_all_mean}')

        else:
            loss_focal = loss_all_mean
            # (BT).

        # FInally, incorporate the remaining half of category-weighted loss values AFTER focal loss.
        loss_final = loss_focal + loss_bias_mean * 0.5

        loss_final = loss_final * w.flatten()
        # (BT).

        return loss_final
