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

import random
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
from nerfstudio.fields.nerfacto_albedo_visibility_field import (
    TCNNRENINerfactoAlbedoField,
)
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.reni_field import get_directions, get_sineweight
from nerfstudio.fields.reni_field_new import RENIField
from nerfstudio.model_components.illumination_samplers import IcosahedronSampler
from nerfstudio.model_components.losses import (
    RENITestLoss,
    RENITestLossMask,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    RGBLambertianRendererWithVisibility,
    RGBRendererWithRENI,
)
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
    reni_loss_mult: float = 1.0
    """Weight for the reni loss"""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    predict_visibility: bool = True
    """Whether to predict visibility or not"""
    visibility_loss_mult: float = 0.01
    """Weight for the visibility loss"""
    illumination_field: Literal["reni", "SH"] = "reni"  # "SH" NOT IMPLEMENTED
    """Illumination field to use"""
    illumination_sampler: Literal["icosahedron", "other"] = "icosahedron"
    """Illumination sampler to use"""
    icosphere_order: int = 2
    """Order of the icosphere to use for illumination sampling"""
    illumination_sample_directions: int = 100
    """Number of directions to sample for illumination"""


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
        self.fitting_eval_latents = False
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

        # illumination sampler
        if self.config.illumination_sampler == "icosahedron":
            self.illumination_sampler = IcosahedronSampler(self.config.icosphere_order)

        # illumination field
        if self.config.illumination_field == "reni":
            self.illumination_field_train = RENIField(self.config.reni_path, num_latent_codes=self.num_train_data)
            self.illumination_field_eval = RENIField(self.config.reni_path, num_latent_codes=self.num_eval_data)

        if self.config.background_model == "grid":
            # overwrite background model with albdeo field
            self.field_background = TCNNRENINerfactoAlbedoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_visibility=self.config.predict_visibility,
            )

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
        self.lambertian_renderer = RGBLambertianRendererWithVisibility()

        if self.config.illumination_field == "reni":
            self.illumination_loss = RENITestLossMask(
                alpha=self.config.reni_prior_loss_weight, beta=self.config.reni_cosine_loss_weight
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["illumination_field"] = list(self.illumination_field_train.parameters())
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

        # Get camera indices for each sample for use in the illumination field
        camera_indices = ray_samples.camera_indices.squeeze()  # [num_rays, samples_per_ray]

        if self.training:
            illumination_field = (
                self.illumination_field_train if not self.fitting_eval_latents else self.illumination_field_eval
            )
        else:
            illumination_field = self.illumination_field_eval

        illumination_directions = self.illumination_sampler(self.config.illumination_sample_directions)
        illumination_directions = illumination_directions.to(self.device)

        field_outputs = self.field(ray_samples, return_alphas=True, illumination_directions=illumination_directions)
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

        # background model
        if self.config.background_model != "none":
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg, True, illumination_directions)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            samples_and_field_outputs["ray_samples_bg"] = ray_samples_bg
            samples_and_field_outputs["field_outputs_bg"] = field_outputs_bg
            samples_and_field_outputs["weights_bg"] = weights_bg

            camera_indices = torch.cat([camera_indices, ray_samples_bg.camera_indices.squeeze()], dim=1)

        # Get illumination for samples
        hdr_light_colours, light_directions = illumination_field(
            camera_indices, None, illumination_directions, "illumination"
        )

        # Get illumination for camera rays
        background_colours, _ = illumination_field(
            camera_indices, None, ray_samples.frustums.directions[:, 0, :], "background"
        )

        samples_and_field_outputs["hdr_light_colours"] = hdr_light_colours
        samples_and_field_outputs["light_directions"] = light_directions
        samples_and_field_outputs["background_colours"] = background_colours

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
        hdr_light_colours = samples_and_field_outputs["hdr_light_colours"]
        light_directions = samples_and_field_outputs["light_directions"]
        background_colours = samples_and_field_outputs["background_colours"]
        albedos = field_outputs[FieldHeadNames.ALBEDO]
        normals = field_outputs[FieldHeadNames.NORMAL]
        visibility = field_outputs[FieldHeadNames.VISIBILITY] if self.config.predict_visibility else None

        if self.config.background_model != "none":
            albedos = torch.cat([albedos, samples_and_field_outputs["field_outputs_bg"][FieldHeadNames.ALBEDO]], dim=1)
            normals = torch.cat(
                [normals, samples_and_field_outputs["field_outputs_bg"][FieldHeadNames.PRED_NORMALS]], dim=1
            )
            visibility = torch.cat(
                [visibility, samples_and_field_outputs["field_outputs_bg"][FieldHeadNames.VISIBILITY]], dim=1
            )
            weights = torch.cat([weights, samples_and_field_outputs["weights_bg"]], dim=1)

        rgb = self.lambertian_renderer(
            albedos=albedos,
            normals=normals,
            light_directions=light_directions,
            light_colors=hdr_light_colours,
            visibility=visibility,
            background_color=background_colours,
            weights=weights,
        )

        albedo = self.renderer_rgb(
            rgb=albedos,
            background_color=torch.ones_like(background_colours),
            weights=weights,
        )

        depth = self.renderer_depth(weights=samples_and_field_outputs["weights"], ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        normal = self.renderer_normal(semantics=normals, weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "albedo": albedo,
            "illumination": background_colours,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "viewdirs": ray_bundle.directions,
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }

        if self.config.predict_visibility:
            outputs["visibility"] = self.renderer_accumulation(weights=visibility)

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

        if self.config.background_model != "none":
            depth_bg = self.renderer_depth(
                weights=samples_and_field_outputs["weights_bg"], ray_samples=samples_and_field_outputs["ray_samples_bg"]
            )
            accumulation_bg = self.renderer_accumulation(weights=samples_and_field_outputs["weights_bg"])

            outputs["depth_bg"] = depth_bg
            outputs["accumulation_bg"] = accumulation_bg

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            if "fg_mask" in batch:
                fg_label = batch["fg_mask"].float().to(self.device)
                # TODO somehow make this not an if statement here
                # Maybe get_loss_dict for the illumination_field should be a function?
                if self.config.illumination_field == "reni":
                    loss_dict["illumination_loss"] = (
                        self.illumination_loss(
                            inputs=outputs["illumination"],
                            targets=batch["image"].to(self.device),
                            mask=fg_label,
                            Z=self.illumination_field_train.get_latents(),
                        )
                        * self.config.reni_loss_mult
                    )

            if self.config.predict_visibility:
                # binary_cross_entropy_with_logits between field_outputs[FieldHeadNames.VISIBILITY] which is shape
                # [K, D, 1] and 1.0 - fg_label which is shape [K, 1] so we need to unsqueeze the fg_label
                loss_dict["visibility_loss"] = (
                    F.binary_cross_entropy_with_logits(
                        outputs["visibility"],
                        1.0 - fg_label,
                    )
                    * self.config.visibility_loss_mult
                )

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
            images_dict["visibility"] = torch.cat([outputs["visibility"]], dim=1)

        return metrics_dict, images_dict

    def fit_latent_codes_for_eval(self, datamanager, gt_source, epochs, learning_rate):
        """Fit evaluation latent codes to session envmaps so that illumination is correct."""
        source = (
            "environment maps"
            if gt_source == "envmap"
            else "sky from left image halves"
            if gt_source == "image_half_sky"
            else "left image halves"
        )
        CONSOLE.print(f"Optimising evaluation latent codes to {source}:")

        opt = torch.optim.Adam(self.illumination_field_eval.parameters(), lr=learning_rate)
        reni_test_loss = RENITestLoss(
            alpha=self.config.reni_prior_loss_weight, beta=self.config.reni_cosine_loss_weight
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[blue]Loss: {task.fields[extra]}"),
        ) as progress:
            task = progress.add_task("[green]Optimising... ", total=epochs, extra="")

            # Setup data source
            if gt_source == "envmap":
                W = datamanager.eval_dataset[0]["envmap"].shape[1]
                directions = get_directions(W)  # [1, H*W, 3] - directions to sample the reni field
                sineweight = get_sineweight(
                    W
                )  # [1, H*W, 3] - sineweight compensation for the irregular sampling of the equirectangular image
            elif gt_source == "image_half_sky":
                sky_colour = torch.tensor([70, 130, 180], dtype=torch.float32).to(self.device) / 255.0
            elif gt_source == "image_half_inverse":
                # Re-initialise latents to zero
                self.illumination_field_eval.reset_latents()
                self.fitting_eval_latents = True

            # Fit latents
            for _ in range(epochs):
                epoch_loss = 0.0
                for step in range(len(datamanager.eval_dataset)):

                    # Lots of admin to get the data in the right format depending on task
                    if gt_source in ["envmap"]:
                        ray_bundle, batch = datamanager.next_eval(step)
                    else:
                        idx, ray_bundle, batch = datamanager.next_eval_image(step)
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
                        rgb = batch["image"].to(self.device)  # [H, W, 3]
                        rgb = rgb[:, : rgb.shape[1] // 2, :]  # [H, W//2, 3]
                        rgb = rgb.reshape(
                            -1, 3
                        )  # [H*W, 3] # TODO: confirm this is row-major (internet seems to think so)

                        # TODO (james): need to mask transients
                        # rebuild a new RayBundle with the left half of the image
                        ray_bundle.origins = ray_bundle.origins[
                            :, : ray_bundle.origins.shape[1] // 2, :, :
                        ]  # [H, W//2, 1, 3]
                        ray_bundle.directions = ray_bundle.directions[
                            :, : ray_bundle.directions.shape[1] // 2, :, :
                        ]  # [H, W//2, 1, 3]
                        ray_bundle.pixel_area = ray_bundle.pixel_area[
                            :, : ray_bundle.pixel_area.shape[1] // 2, :, :
                        ]  # [H, W//2, 1, 1]
                        ray_bundle.directions_norm = ray_bundle.directions_norm[
                            :, : ray_bundle.directions_norm.shape[1] // 2, :, :
                        ]  # [H, W//2, 1, 1]
                        ray_bundle.camera_indices = ray_bundle.camera_indices[
                            :, : ray_bundle.camera_indices.shape[1] // 2, :, :
                        ]  # [H, W//2, 1, 1]

                        # TODO handle nears and fars if they are defined and meta data and times.
                        ray_bundle = ray_bundle.get_row_major_sliced_ray_bundle(0, len(ray_bundle))  # [H*W, N]
                        # ray_bundle = ray_bundle.sample(512) # this doesn't work, not sure why TODO
                        # sample N rays and build new ray_bundle
                        indices = random.sample(range(len(ray_bundle)), k=256)
                        ray_bundle.origins = ray_bundle.origins[indices, :]  # [N, 3]
                        ray_bundle.directions = ray_bundle.directions[indices, :]  # [N, 3]
                        ray_bundle.pixel_area = ray_bundle.pixel_area[indices, :]  # [N, 1]
                        ray_bundle.directions_norm = ray_bundle.directions_norm[indices, :]  # [N, 1]
                        ray_bundle.camera_indices = ray_bundle.camera_indices[indices, :]  # [N, 1]

                        # also get rgb values for the sampled rays
                        rgb = rgb[indices, :]  # [N, 3]

                    # Get model output
                    # TODO Fix these two: they are not working
                    if gt_source in ["envmap", "image_half_sky"]:
                        # sample the reni field
                        Z = self.illumination_field_eval.get_Z()[idx, :, :]
                        if len(Z.shape) == 2:
                            Z = Z.unsqueeze(0)

                        D = torch.stack([-D[:, :, 0], D[:, :, 2], D[:, :, 1]], dim=2)  # [B, num_rays, 3]
                        model_output = self.illumination_field_eval(Z, D)  # [B, num_rays, 3]
                        model_output = self.illumination_field_eval.unnormalise(
                            model_output
                        )  # undo reni scaling between -1 and 1
                        model_output = sRGB(model_output)  # undo reni gamma correction
                    else:
                        outputs = self.forward(ray_bundle=ray_bundle)
                        model_output = outputs["rgb"]  # [N, 3]

                    opt.zero_grad()
                    if gt_source in ["envmap", "image_half_sky"]:
                        loss, _, _, _ = reni_test_loss(model_output, rgb, S, Z)
                    else:
                        loss = (
                            self.rgb_loss(rgb, model_output)
                            + self.config.reni_prior_loss_weight
                            * torch.pow(self.illumination_field_eval.get_latents(), 2).sum()
                        )
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()

                progress.update(task, advance=1, extra=f"{epoch_loss:.4f}")

        if gt_source in ["envmap", "image_half"]:
            self.illumination_field_eval.set_no_grad()  # We no longer need to optimise latent codes as done prior to start of training

        # no longer fitting latents
        self.fitting_eval_latents = False
