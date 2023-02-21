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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import BarColumn, Console, Progress, TextColumn, TimeRemainingColumn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.reni_field import get_directions, get_reni_field, get_sineweight
from nerfstudio.model_components.losses import (
    MSELoss,
    RENITestLoss,
    RENITestLossMask,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import sRGB

CONSOLE = Console(width=120)


@dataclass
class RENINerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: RENINerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 48
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
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = True
    """Whether to predict normals or not."""
    reni_path: str = "path/to/checkpoint.ckpt"
    """Path to pretrained reni model"""
    icosphere_order: int = 2
    """Order of the icosphere for the reni model"""
    reni_prior_loss_weight: float = 1e-7
    """Weight for the reni prior loss"""
    reni_cosine_loss_weight: float = 1e-1
    """Weight for the reni cosine loss"""
    fg_mask_loss_mult: float = 0.01
    """Weight for the foreground mask loss"""
    reni_loss_weight: float = 1.0
    """Weight for the reni loss"""
    use_visibility: bool = True
    """Whether to predict visibility or not"""
    visibility_loss_mult: float = 0.01
    """Weight for the visibility loss"""


class RENINerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: RENINerfactoModelConfig

    def __init__(
        self,
        config: RENINerfactoModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        num_eval_data: int,
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
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

        scene_contraction = SceneContraction(order=float("inf"))

        self.reni_field_train = get_reni_field(self.config.reni_path, num_latent_codes=self.num_train_data)
        self.reni_field_eval = get_reni_field(self.config.reni_path, num_latent_codes=self.num_eval_data)

        # Fields
        self.field = TCNNRENINerfactoAlbedoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            icosphere_order=self.config.icosphere_order,
            use_visibility=self.config.use_visibility,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        self.renderer_rgb = RGBRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.reni_loss = RENITestLossMask(
            alpha=self.config.reni_prior_loss_weight, beta=self.config.reni_cosine_loss_weight
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        param_groups["reni_field"] = list(self.reni_field_train.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
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

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        reni = self.reni_field_train if self.training else self.reni_field_eval

        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals, reni_field=reni)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

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
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "albedo": albedo,
            "illumination": field_outputs[FieldHeadNames.ILLUMINATION],
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.use_visibility:
            outputs["accumulated_visibility"] = self.renderer_accumulation(
                weights=field_outputs[FieldHeadNames.VISIBILITY]
            )
            outputs["visibility"] = field_outputs[FieldHeadNames.VISIBILITY]

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if True or self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # check if nans in outputs
        for k, v in outputs.items():
            # check if its a tensor and if it has any nans
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                raise ValueError(f"NaNs in output {k}")

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
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

                # foreground mask loss
                if self.config.fg_mask_loss_mult > 0.0:
                    w = outputs["weights_list"][-1]
                    weights_sum = w.sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                    loss_dict["fg_mask_loss"] = (
                        F.binary_cross_entropy_with_logits(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                    )

                if self.config.use_visibility:
                    # binary_cross_entropy_with_logits between field_outputs[FieldHeadNames.VISIBILITY] which is shape
                    # [K, D, 1] and 1.0 - fg_label which is shape [K, 1] so we need to unsqueeze the fg_label
                    loss_dict["visibility_loss"] = (
                        F.binary_cross_entropy_with_logits(
                            outputs["visibility"],
                            1.0 - fg_label.unsqueeze(1).repeat(1, outputs["visibility"].shape[1], 1),
                        )
                        * self.config.visibility_loss_mult
                    )

            else:
                loss_dict["reni_loss"] = (
                    self.config.reni_prior_loss_weight * torch.pow(self.reni_field.get_Z(), 2).sum()
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        albedo = outputs["albedo"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_albedo = torch.cat([albedo], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "albedo": combined_albedo,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        images_dict["illumination"] = torch.cat([outputs["illumination"]], dim=1)
        images_dict["albedo"] = torch.cat([outputs["albedo"]], dim=1)

        if "visibility" in outputs:
            images_dict["visibility"] = torch.cat([outputs["accumulated_visibility"]], dim=1)

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def fit_latent_codes_for_eval(self, eval_dataset):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""
        CONSOLE.print("Fitting RENI evaluation latent codes to envmaps...")

        epochs = 150
        opt = torch.optim.Adam(self.reni_field_eval.parameters(), lr=1e-2)
        criterion = RENITestLoss(alpha=1e-9, beta=1e-1)

        W = eval_dataset[0]["envmap"].shape[1]
        # H = W // 2

        # # directions to sample the reni field
        directions = get_directions(W)  # [1, H*W, 3]
        # sineweight compensation for the irregular sampling of the equirectangular image
        sineweight = get_sineweight(W)  # [1, H*W, 3]

        iter_dataset = iter(eval_dataset)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("Loss: {task.fields[extra]}"),
        ) as progress:
            task = progress.add_task("[green]Fitting...", total=epochs, extra="")
            for _ in range(epochs):
                epoch_loss = 0.0
                for _ in range(len(eval_dataset)):
                    try:
                        batch = next(iter_dataset)
                    except StopIteration:
                        iter_dataset = iter(eval_dataset)
                        batch = next(iter_dataset)
                    assert batch["envmap"] != "none"
                    idx = batch["image_idx"]
                    envmap = batch["envmap"].to(self.device)  # [H, W, 3]
                    envmap = envmap.unsqueeze(0)  # [1, H, W, 3]
                    envmap = envmap.reshape(1, -1, 3)  # [1, H*W, 3]
                    mask = batch["envmap_fg_mask"].to(self.device)  # [H, W]
                    mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
                    mask = mask.repeat(1, 1, 1, 3)  # [1, H, W, 3]
                    mask = mask.reshape(1, -1, 3)  # [1, H*W, 3]

                    D = directions.type_as(envmap)
                    S = sineweight.type_as(envmap)
                    S = S * mask

                    Z = self.reni_field_eval.get_Z()[idx, :, :].unsqueeze(0)  # [1, ndims, 3]

                    model_output = self.reni_field_eval(Z, D)  # [1, H*W, 3]
                    model_output = self.reni_field_eval.unnormalise(model_output)  # [1, H*W, 3]
                    model_output = sRGB(model_output)  # [1, H*W, 3]

                    opt.zero_grad()
                    loss, _, _, _ = criterion(model_output, envmap, S, Z)
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()

                progress.update(task, advance=1, extra=epoch_loss)

        self.reni_field_eval.mu.requires_grad = False
