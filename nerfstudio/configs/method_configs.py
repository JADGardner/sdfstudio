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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    MachineConfig,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    FlexibleDataManagerConfig,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSRDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialSchedulerConfig,
    MultiStepSchedulerConfig,
    NeuSSchedulerConfig,
)
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_albedo_visibility_field import SDFAlbedoVisibilityFieldConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.dto import DtoOModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neuralreconW import NeuralReconWModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_acc import NeuSAccModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.reni_nerfacto import RENINerfactoModelConfig
from nerfstudio.models.reni_neus import RENINeuSModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.unisurf import UniSurfModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.volsdf import VolSDFModelConfig
from nerfstudio.pipelines.base_pipeline import (
    FlexibleInputPipelineConfig,
    VanillaPipelineConfig,
)
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.pipelines.fit_eval_latents_pipeline import FitEvalLatentsPipelineConfig

method_configs: Dict[str, Config] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for bounded synthetic data.",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "volsdf": "Implementation of VolSDF.",
    "monosdf": "Implementation of MonoSDF.",
    "geo-volsdf": "Implementation of patch warping from GeoNeuS with VolSDF.",
    "neus": "Implementation of NeuS.",
    "mono-neus": "Implementation of MonoSDF with NeuS rendering formulation.",
    "geo-neus": "Implementation of patch warping from GeoNeuS with NeuS.",
    "unisurf": "Implementation of UniSurf.",
    "mono-unisurf": "Implementation of MonoSDF with unisurf rendering formulation.",
    "geo-unisurf": "Implementation of patch warping from GeoNeuS with UniSurf.",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "dto": "Occupancy field with density guided sampling",
    "neusW": "Implementation of Neural Reconstruction in the wild",
    "neus-acc": "Implementation of NeuS with empty space skipping.",
    "neus-facto": "Implementation of NeuS similar to nerfacto where proposal sampler is used.",
    "neus-facto-bigmlp": "NeuS-facto with big MLP, it is used in training heritage data with 8 gpus",
    "RENI-NeuS": "NeuS-facto with illumination, albedo and shadow disentanglement",
    "RENI-Nerfacto": "Nerfacto with illumination, albedo and shadow disentanglement",
}

method_configs["neus-facto"] = Config(
    method_name="neus-facto",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_grid_feature=True,
                num_layers=5,
                num_layers_color=5,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=True,
                inside_outside=False,
            ),
            background_model="mlp",
            eval_num_rays_per_chunk=1024,
            near_plane=0.05,
            far_plane=4.0,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus-facto-bigmlp"] = Config(
    method_name="neus-facto-bigmlp",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100001,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(num_layers=8, hidden_dim=512, num_layers_color=4), eval_num_rays_per_chunk=1024
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=100000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["geo-volsdf"] = Config(
    method_name="geo-volsdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200001,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(
                max_steps=1000000
            ),  # set max_steps to a large value so it never step and we will use the last_lr form the pretrained model
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["monosdf"] = Config(
    method_name="monosdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["volsdf"] = Config(
    method_name="volsdf",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=VolSDFModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=100000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=100000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["geo-neus"] = Config(
    method_name="geo-neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=200000,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["mono-neus"] = Config(
    method_name="mono-neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neus"] = Config(
    method_name="neus",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["unisurf"] = Config(
    method_name="unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["mono-unisurf"] = Config(
    method_name="mono-unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(mono_depth_loss_mult=0.1, mono_normal_loss_mult=0.05, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["geo-unisurf"] = Config(
    method_name="geo-unisurf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=FlexibleInputPipelineConfig(
        datamanager=FlexibleDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=UniSurfModelConfig(patch_warp_loss_mult=0.1, eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["dto"] = Config(
    method_name="dto",
    trainer=TrainerConfig(
        steps_per_eval_image=2000,
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=DtoOModelConfig(eval_num_rays_per_chunk=1 << 10),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "occupancy_field": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neusW"] = Config(
    method_name="neusW",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuralReconWModelConfig(
            background_model="grid", num_samples_outside=4, eikonal_loss_mult=0.0001, eval_num_rays_per_chunk=1 << 10
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

method_configs["neus-acc"] = Config(
    method_name="neus-acc",
    trainer=TrainerConfig(
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSAccModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


method_configs["nerfacto"] = Config(
    method_name="nerfacto",
    trainer=TrainerConfig(
        steps_per_eval_batch=5000, steps_per_save=2000, max_num_iterations=30000, mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = Config(
    method_name="instant-ngp",
    trainer=TrainerConfig(
        steps_per_eval_batch=5000,
        steps_per_eval_image=5000,
        steps_per_save=20000,
        max_num_iterations=20001,
        mixed_precision=True,
        steps_per_eval_all_images=20000,
    ),
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(render_step_size=0.005, eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20000),
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

method_configs["mipnerf"] = Config(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = Config(
    method_name="semantic-nerfw",
    trainer=TrainerConfig(
        steps_per_eval_batch=500, steps_per_save=2000, max_num_iterations=30000, mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=FriendsDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = Config(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = Config(
    method_name="tensorf",
    trainer=TrainerConfig(mixed_precision=False),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
)

method_configs["dnerf"] = Config(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["phototourism"] = Config(
    method_name="phototourism",
    trainer=TrainerConfig(
        steps_per_eval_batch=500, steps_per_save=2000, max_num_iterations=30000, mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VariableResDataManagerConfig(  # NOTE: one of the only differences with nerfacto
            dataparser=PhototourismDataParserConfig(),  # NOTE: one of the only differences with nerfacto
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["RENI-Nerfacto"] = Config(
    method_name="RENI-Nerfacto",
    trainer=TrainerConfig(
        steps_per_eval_batch=5000, steps_per_save=2000, max_num_iterations=30000, mixed_precision=True
    ),
    pipeline=FitEvalLatentsPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=RENINerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            reni_path="checkpoints/reni_pretrained_weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1579.ckpt",
            icosphere_order=3,
            fg_mask_loss_mult=1.0,
            reni_loss_weight=1.0,
            far_plane=20.0,
            use_visibility=True,
            visibility_loss_mult=1.0,
            # reni_cosine_loss_weight=0.1,
            # reni_prior_loss_weight=0.1,
            # num_nerf_samples_per_ray=12,
            # predict_normals=False,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
            # "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
            # "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=300000),
        },
        "reni_field": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": SchedulerConfig(lr_final=1e-3, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["RENI-NeuS"] = Config(
    method_name="RENI-NeuS",
    machine=MachineConfig(
        num_gpus=1,
    ),
    trainer=TrainerConfig(
        steps_per_eval_image=10000,
        steps_per_eval_batch=100000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=120001,
        mixed_precision=False,
        # load_dir="outputs/data-NeRF-OSR-Data/RENI-NeuS/latest_with_rot_and_clip_illumination/sdfstudio_models",
        # load_step=100000,
    ),
    pipeline=FitEvalLatentsPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NeRFOSRDataParserConfig(),
            train_num_rays_per_batch=256,
            eval_num_rays_per_batch=256,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=RENINeuSModelConfig(
            sdf_field=SDFAlbedoVisibilityFieldConfig(
                use_grid_feature=True,
                num_layers=5,
                num_layers_color=5,
                num_layers_visibility=2,
                hidden_dim=256,
                hidden_dim_color=256,
                hidden_dim_visibility=64,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,
                inside_outside=False,  # False is for outdoor scenes
                use_visibility="none",
            ),
            eval_num_rays_per_chunk=256,
            fg_mask_loss_mult=1.0,
            reni_loss_mult=1.0,
            reni_path="checkpoints/reni_pretrained_weights/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_decoder_epoch=1579.ckpt",
            train_reni_scale=True,
            near_plane=0.05,
            far_plane=4.0,
            albedo_scale=1.0,
            far_plane_bg=1000.0,
            visibility_loss_mse_mult=1.0,
            background_model="none",
            use_average_appearance_embedding=False,
            icosphere_order=11,
            illumination_sampler_random_rotation=True,
            illumination_sample_remove_lower_hemisphere=False,
        ),
        eval_latent_optimisation_source="image_half_inverse",
        eval_latent_optimisation_epochs=50,
        eval_latent_optimisation_lr=1e-2,
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=1.0, max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=1.0, max_steps=100000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": SchedulerConfig(lr_final=1e-3, max_steps=100000),
        },
        "illumination_field": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=100000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
