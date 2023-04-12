# set the cwd to the root of the repo

import os

os.chdir("/users/jadg502/scratch/code/sdfstudio/")

import torch
import yaml
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.nerfosr_dataparser import NeRFOSR, NeRFOSRDataParserConfig
from nerfstudio.models.reni_neus import RENINeuSModel, RENINeuSModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaDataManager
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.reni_field import get_directions
from nerfstudio.fields.reni_field import RENIField, get_directions, get_sineweight
from nerfstudio.cameras.rays import RayBundle

print(torch.cuda.is_available())


def make_ray_bundle_copy(ray_bundle):
    new_ray_bundle = RayBundle(
        origins=ray_bundle.origins.detach().clone(),
        directions=ray_bundle.directions.detach().clone(),
        pixel_area=ray_bundle.pixel_area.detach().clone(),
        directions_norm=ray_bundle.directions_norm.detach().clone(),
        camera_indices=ray_bundle.camera_indices.detach().clone(),
        nears=ray_bundle.nears.detach().clone() if ray_bundle.nears is not None else None,
        fars=ray_bundle.fars.detach().clone() if ray_bundle.fars is not None else None,
    )
    return new_ray_bundle


def make_batch_clone(batch):
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            new_batch[key] = value.detach().clone()
        else:
            new_batch[key] = value
    return new_batch


def sRGB(imgs):
    # if shape is not B, C, H, W, then add batch dimension
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    q = torch.quantile(torch.quantile(torch.quantile(imgs, 0.98, dim=(1)), 0.98, dim=(1)), 0.98, dim=(1))
    imgs = imgs / q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    imgs = torch.clamp(imgs, 0.0, 1.0)
    imgs = torch.where(
        imgs <= 0.0031308,
        12.92 * imgs,
        1.055 * torch.pow(torch.abs(imgs), 1 / 2.4) - 0.055,
    )
    return imgs


import numpy as np


def rotation_matrix(axis, angle):
    """
    Return 3D rotation matrix for rotating around the given axis by the given angle.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


# setup config
test_mode = "val"
world_size = 1
local_rank = 1
device = "cuda:1"

ckpt_path = "outputs/data-NeRF-OSR-Data/RENI-NeuS/latest_with_rot_and_clip_illumination/"
step = 100000

ckpt = torch.load(ckpt_path + "/sdfstudio_models" + f"/step-{step:09d}.ckpt", map_location=device)
model_dict = {}
for key in ckpt["pipeline"].keys():
    if key.startswith("_model."):
        model_dict[key[7:]] = ckpt["pipeline"][key]

# load yaml checkpoint config
config_path = Path(ckpt_path) / "config.yml"
config = yaml.load(config_path.open(), Loader=yaml.Loader)

pipeline_config = config.pipeline
pipeline_config.datamanager.dataparser.scene = "lk2"
pipeline_config.datamanager.dataparser.use_session_data = False

# if illumination_sampler_random_rotation not in pipeline.config.model add it and set to false
try:
    pipeline_config.model.illumination_sampler_random_rotation
except AttributeError:
    pipeline_config.model.illumination_sampler_random_rotation = True
try:
    pipeline_config.model.illumination_sample_remove_lower_hemisphere
except AttributeError:
    pipeline_config.model.illumination_sample_remove_lower_hemisphere = True

datamanager: VanillaDataManager = pipeline_config.datamanager.setup(
    device=device,
    test_mode=test_mode,
    world_size=world_size,
    local_rank=local_rank,
)
datamanager.to(device)
# includes num_eval_data as needed for reni latent code fitting.
model = pipeline_config.model.setup(
    scene_box=datamanager.train_dataset.scene_box,
    num_train_data=len(datamanager.train_dataset),
    num_eval_data=len(datamanager.eval_dataset),
    metadata=datamanager.train_dataset.metadata,
    world_size=world_size,
    local_rank=local_rank,
    eval_latent_optimisation_source=pipeline_config.eval_latent_optimisation_source,
)
model.to(device)

model.load_state_dict(model_dict)
model.eval()

image_idx_original, camera_ray_bundle_original, batch_original = datamanager.next_eval_image(1)

reni_field = RENIField(pipeline_config.model.reni_path, num_latent_codes=1673, fixed_decoder=False)
reni = reni_field.reni
reni.fixed_decoder = True

Z = torch.load("checkpoints/reni_pretrained_weights/z_point_light.pt")
Z = Z.repeat(model.num_eval_data, 1, 1).to(device)
model.illumination_field_eval.reni.mu.data = Z

model.use_visibility = "sdf"
# model.icosphere_order = 11
camera_ray_bundle = make_ray_bundle_copy(camera_ray_bundle_original)
batch = make_batch_clone(batch_original)

camera_ray_bundle.nears = torch.zeros_like(camera_ray_bundle.directions_norm) + model.scene_box.near

camera_ray_bundle.fars = torch.zeros_like(camera_ray_bundle.directions_norm) + model.scene_box.far

model.config.eval_num_rays_per_chunk = 2048

outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, show_progress=True)

# save outputs[rgb]
plt.imsave("rgb.png", outputs["rgb"].cpu().detach().numpy())
