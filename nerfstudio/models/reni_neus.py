# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.reni_field import get_directions, get_reni_field, get_sineweight
from nerfstudio.model_components.losses import (
    RENITestLoss,
    RENITestLossMask,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import RGBRendererWithRENI
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import sRGB

CONSOLE = Console(width=120)


@dataclass
class RENINeuSModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: RENINeuSModel)
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_neus_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    reni_path: str = "path/to/checkpoint.ckpt"
    """Path to pretrained reni model"""
    reni_prior_loss_weight: float = 1e-7
    """Weight for the reni prior loss"""
    reni_cosine_loss_weight: float = 1e-1
    """Weight for the reni cosine loss"""
    reni_loss_weight: float = 1.0
    """Weight for the reni loss"""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    predict_visibility: bool = True
    """Whether to predict visibility or not"""
    visibility_loss_mult: float = 0.01
    """Weight for the visibility loss"""


class RENINeuSModel(NeuSModel):
    """NeuS facto model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: RENINeuSModelConfig

    def __init__(
        self,
        config: RENINeuSModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        num_eval_data: int,
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ) -> None:
        self.num_eval_data = num_eval_data  # needed for fitting latent codes to envmaps for eval data
        super().__init__(
            config=config,
            scene_box=scene_box,
            num_train_data=num_train_data,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.use_average_appearance_embedding = False

        self.reni_field_train = get_reni_field(self.config.reni_path, num_latent_codes=self.num_train_data)
        self.reni_field_eval = get_reni_field(self.config.reni_path, num_latent_codes=self.num_eval_data)

        # try:
        #     self.field.set_reni_field(self.reni_field)
        # except AttributeError as exc:
        #     raise AttributeError("Must use the reni_sdf_albedo field") from exc

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # update proposal network every iterations
        update_schedule = lambda step: -1

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            use_uniform_sampler=True,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        self.renderer_rgb = RGBRendererWithRENI()
        self.reni_loss = RENITestLossMask(
            alpha=self.config.reni_prior_loss_weight, beta=self.config.reni_cosine_loss_weight
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["field_background"] = list(self.reni_field_train.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        reni = self.reni_field_train if self.training else self.reni_field_eval

        field_outputs = self.field(ray_samples, return_alphas=True, reni_field=reni)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
        }
        return samples_and_field_outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        # TODO make this configurable
        # compute near and far from from sphere with radius 1.0
        # ray_bundle = self.sphere_collider(ray_bundle)

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        # bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RADIANCE],
            background_color=field_outputs[FieldHeadNames.ILLUMINATION],
            weights=weights,
        )

        albedo = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.ALBEDO],
            background_color=torch.ones_like(field_outputs[FieldHeadNames.ILLUMINATION]),
            weights=weights,
        )

        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "albedo": albedo,
            "illumination": field_outputs[FieldHeadNames.ILLUMINATION],
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            # "normals": field_outputs[FieldHeadNames.NORMAL],
            "weights": weights,
            "viewdirs": ray_bundle.directions,
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }

        if self.config.predict_visibility:
            outputs["accumulated_visibility"] = self.renderer_accumulation(
                weights=field_outputs[FieldHeadNames.VISIBILITY]
            )
            outputs["visibility"] = field_outputs[FieldHeadNames.VISIBILITY]

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0

        # check if nans in outputs
        for k, v in outputs.items():
            # check if its a tensor and if it has any nans
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                raise ValueError(f"NaNs in output {k}")

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            if "fg_mask" in batch:
                fg_label = batch["fg_mask"].float().to(self.device)
                loss_dict["reni_loss"] = (
                    self.reni_loss(
                        inputs=outputs["illumination"],
                        targets=batch["image"].to(self.device),
                        mask=fg_label,
                        Z=self.reni_field_train.get_Z(),
                    )
                    * self.config.reni_loss_weight
                )

            if self.config.predict_visibility:
                # binary_cross_entropy_with_logits between field_outputs[FieldHeadNames.VISIBILITY] which is shape
                # [K, D, 1] and 1.0 - fg_label which is shape [K, 1] so we need to unsqueeze the fg_label
                loss_dict["visibility_loss"] = (
                    F.binary_cross_entropy_with_logits(
                        outputs["visibility"],
                        1.0 - fg_label.unsqueeze(1).repeat(1, outputs["visibility"].shape[1], 1),
                    )
                    * self.config.visibility_loss_mult
                )
                # loss_dict["visibility_loss"] = F.binary_cross_entropy_with_logits(outputs["visibility"], 1.0 - fg_label)

            # loss_dict["orientation_loss"] = self.config.orientation_loss_mult * orientation_loss(
            #     weights=outputs["weights"], normals=outputs["normals"], viewdirs=outputs["viewdirs"]
            # )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        images_dict["illumination"] = torch.cat([outputs["illumination"]], dim=1)
        images_dict["albedo"] = torch.cat([outputs["albedo"]], dim=1)

        if "visibility" in outputs:
            images_dict["visibility"] = torch.cat([outputs["accumulated_visibility"]], dim=1)

        return metrics_dict, images_dict

    def fit_latent_codes_for_eval(self, datamanager, gt_source):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""
        CONSOLE.print("Fitting RENI evaluation latent codes to envmaps...")

        # TODO Make configurable
        epochs = 30
        opt = torch.optim.Adam(self.reni_field_eval.parameters(), lr=1e-1)
        criterion = RENITestLoss(alpha=1e-9, beta=1e-1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[extra]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising eval latents...", total=epochs, extra="")

            # Setup data source
            if gt_source == "envmap":
                dataloader = datamanager.eval_dataloader  # for loading batch of rays
                W = datamanager.eval_dataset[0]["envmap"].shape[1]
                directions = get_directions(W)  # [1, H*W, 3] - directions to sample the reni field
                sineweight = get_sineweight(
                    W
                )  # [1, H*W, 3] - sineweight compensation for the irregular sampling of the equirectangular image
            elif gt_source == "image_half_sky":
                dataloader = datamanager.fixed_indices_eval_dataloader  # for loading full images
                sky_colour = torch.tensor([70, 130, 180], dtype=torch.float32).to(self.device) / 255.0
            elif gt_source == "image_half_inverse":
                dataloader = datamanager.eval_dataloader  # for loading batch of rays
                # Set latent codes to zeros

            # Fit latents
            for _ in range(epochs):
                epoch_loss = 0.0
                for ray_bundle, batch in dataloader:
                    idx = batch["image_idx"]
                    if gt_source == "envmap":
                        rgb = batch["envmap"].to(self.device)
                        rgb = rgb.unsqueeze(0)  # [B, H, W, 3]
                        rgb = rgb.reshape(rgb.shape[0], -1, 3)  # [B, H*W, 3]
                        mask = batch["envmap_fg_mask"].to(self.device)  # [B, H, W]
                        mask = mask.unsqueeze(-1)  # [B, H, W, 1]
                        mask = mask.repeat(1, 1, 1, 3)  # [1, H, W, 3]
                        mask = mask.reshape(mask.shape[0], -1, 3)  # [1, H*W, 3]
                        D = directions.type_as(rgb).repeat(rgb.shape[0], 1, 1)  # [B, H*W, 3]
                        S = sineweight.type_as(rgb).repeat(rgb.shape[0], 1, 1)  # [B, H*W, 3]
                        S = S * mask
                    elif gt_source == "image_half_sky":
                        directions = ray_bundle.directions.to(self.device)  # [H, W, 3]
                        rgb = batch["image"].images[0].to(self.device)  # [H, W, 3]
                        semantic_mask = batch["semantic_mask"].images[0].to(self.device)  # [H, W, 3]

                        # take just the left half of the image
                        rgb = rgb[:, : rgb.shape[1] // 2, :]  # [H, W//2, 3]
                        directions = directions[:, : directions.shape[1] // 2, :]  # [H, W//2, 3]
                        semantic_mask = semantic_mask[:, : semantic_mask.shape[1] // 2, :]  # [H, W//2, 3]

                        # select only rgb and directions for semantic mask sky
                        rgb = rgb[torch.all(torch.eq(semantic_mask, sky_colour), dim=2)].unsqueeze(
                            0
                        )  # [1, num_sky_rays, 3]
                        D = directions[torch.all(torch.eq(semantic_mask, sky_colour), dim=2)].unsqueeze(
                            0
                        )  # [1, num_sky_rays, 3]
                        S = torch.ones_like(rgb)  # [1, num_sky_rays, 3] # no need for sineweight compensation here
                    elif gt_source == "image_half_inverse":
                        rgb = batch["image"].to(self.device)  # [num_eval_rays, 3]
                        directions = ray_bundle.directions.to(self.device)  # [num_eval_rays, 3]

                    # sample the reni field
                    Z = self.reni_field_eval.get_Z()[idx, :, :]
                    if len(Z.shape) == 2:
                        Z = Z.unsqueeze(0)

                    D = torch.stack([-D[:, :, 0], D[:, :, 2], D[:, :, 1]], dim=2)  # [B, num_rays, 3]
                    model_output = self.reni_field_eval(Z, D)  # [B, num_rays, 3]
                    model_output = self.reni_field_eval.unnormalise(model_output)  # undo reni scaling between -1 and 1
                    model_output = sRGB(model_output)  # undo reni gamma correction
                    opt.zero_grad()
                    loss, _, _, _ = criterion(model_output, rgb, S, Z)
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()

                progress.update(task, extra=f"{epoch_loss:.4f}")

        if gt_source in ["envmap", "image_half"]:
            self.reni_field_eval.mu.requires_grad = (
                False  # We no longer need to optimise latent codes as done prior to start of training
            )
