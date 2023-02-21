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

"""Data parser for NeRF-OSR dataset"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RANSACRegressor
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json

# from nerfstudio.utils.colors import get_color


def get_masks(image_idx: int, masks, fg_masks, semantic_masks):
    """function to process additional mask information

    Args:
        image_idx: specific image index to work with
        mask: mask data
    """

    # mask
    mask = masks[image_idx]
    mask = BasicImages([mask])

    # foreground mask
    fg_mask = fg_masks[image_idx]
    fg_mask = BasicImages([fg_mask])

    # semantic mask
    semantic_mask = semantic_masks[image_idx]
    semantic_mask = BasicImages([semantic_mask])

    return {"mask": mask, "fg_mask": fg_mask, "semantic_mask": semantic_mask}


def get_session_data(image_idx: int, capture_sessions):
    """function to process additional session specific data

    Args:
        image_idx: specific image index to work with
        capture_sessions: capture session data
    """

    session = capture_sessions["image_idx_to_session"][image_idx]
    envmap = capture_sessions["sessions"][session]["envmap"]
    envmap_semantics = capture_sessions["sessions"][session]["envmap_semantics"]
    envmap_fg_mask = capture_sessions["sessions"][session]["envmap_fg_mask"]

    return {
        "session": session,
        "envmap": envmap,
        "envmap_semantics": envmap_semantics,
        "envmap_fg_mask": envmap_fg_mask,
    }


# def estimate_object_position_offset(cameras: Cameras):
#     """Estimate the location of the object rays from centre of cameras."""

#     # camera_indices: Union[TensorType["num_rays":..., "num_cameras_batch_dims"], int],
#     # coords: Optional[TensorType["num_rays":..., 2]] = None,
#     cx = cameras.cx
#     cy = cameras.cy
#     coords = torch.stack([cx, cy], dim=-1).squeeze()
#     indices = torch.arange(cameras.camera_to_worlds.shape[0], device=cameras.device).unsqueeze(0)
#     raybundle = cameras.generate_rays(camera_indices=indices, coords=coords)


def rotation_matrix_from_vectors(vec1, vec2):
    """Compute the rotation matrix that aligns vec1 to vec2.

    Args:
    vec1 (torch.Tensor): shape (3,), the starting vector
    vec2 (torch.Tensor): shape (3,), the ending vector

    Returns:
    torch.Tensor: shape (3, 3), the rotation matrix that aligns vec1 to vec2
    """
    a, b = (vec1 / vec1.norm(dim=1)).squeeze(), (vec2 / vec2.norm(dim=1)).squeeze()
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = v.norm()
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + kmat + torch.mm(kmat, kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def get_world_to_z_up(c2w):
    """Get the rotation matrix that rotates an arbitray world coordinate
    system to the coordinate system with the z-axis pointing up.

    Args:
        c2w: a 4x4 camera-to-world transformation matrix

    Returns:
        w2yup: a 4x4 world-to-Zup transformation matrix
        plane: a 3-tuple (a, b, d) of plane parameters
    """
    t = c2w[:, :3, 3]  # position of camera centers in world coordinates

    # fit a plane to the camera centers
    ransac = RANSACRegressor().fit(t[:, :2].cpu().detach().numpy(), t[:, 2].cpu().detach().numpy())

    a, b = ransac.estimator_.coef_  # coefficients
    d = ransac.estimator_.intercept_  # intercept

    # get plane normal
    plane_normal = torch.tensor([a, b, -1]).reshape(1, 3)
    plane_normal = plane_normal / plane_normal.norm()

    # TODO I've set +1.0 to get the model pointing up in the
    #      +z direction, but this might not work for all cases??
    rotation_to_z_up = rotation_matrix_from_vectors(plane_normal, torch.tensor([0.0, 0.0, -1.0]).reshape(1, 3))

    w2zup = torch.eye(4)
    w2zup[:3, :3] = rotation_to_z_up

    return w2zup, (a, b, d)


def find_files(directory, exts):
    """Find all files in a directory that have a certain file extension.

    Parameters
    ----------
    directory : str
        The directory to search for files.
    exts : list of str
        A list of file extensions to search for. Each file extension should be in the form '*.ext'.

    Returns
    -------
    list of str
        A list of file paths for all the files that were found. The list is sorted alphabetically.
    """
    if os.path.isdir(directory):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(directory, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    return []


@dataclass
class NeRFOSRDataParserConfig(DataParserConfig):
    """NeRF-OSR dataset parser config"""

    _target: Type = field(default_factory=lambda: NeRFOSR)
    """target class to instantiate"""
    data: Path = Path("data/NeRF-OSR/Data/")
    """Directory specifying location of data."""
    scene: str = "stjacob"
    """which scene to load"""
    scene_scale: float = 3.0
    """How much to scale the region of interest by."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    use_masks: bool = True
    """Whether to use masks."""
    masks_from_semantics: bool = True
    """Use semantic classes as masks rather than original masks."""
    verbose: bool = False
    """Load dataset with verbose messaging"""
    align_poses_method: Literal["ransac", "pca", "up", "none"] = "ransac"
    """Whether to align the poses to the Z-up coordinate system."""
    center_poses: bool = True
    """Whether to center the poses around the origin."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    use_session_data: bool = True
    """Whether to use return session data and environment maps."""
    environment_map_resolution: int = 256
    """Width of environment maps."""


@dataclass
class NeRFOSR(DataParser):
    """NeRFOSR Dataset
    Some of this code comes from https://github.com/r00tman/NeRF-OSR/blob/main/data_loader_split.py

    Source data convention is:
    camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
    poses is camera-to-world
    masks are 0 for dynamic content, 255 for static content
    """

    config: NeRFOSRDataParserConfig

    def __init__(self, config: NeRFOSRDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scene: str = config.scene
        self.scene_dir = f"{self.data}/{self.scene}/final"

    def _generate_dataparser_outputs(self, split="train"):
        def parse_txt(filename):
            assert os.path.isfile(filename)
            nums = open(filename, encoding="UTF-8").read().split()
            return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

        def get_camera_params(split):
            split = "validation" if split == "val" else split
            split_dir = f"{self.scene_dir}/{split}"

            # camera parameters files
            intrinsics_files = find_files(f"{split_dir}/intrinsics", exts=["*.txt"])
            pose_files = find_files(f"{split_dir}/pose", exts=["*.txt"])

            num_cams = len(pose_files)

            intrinsics = []
            camera_to_worlds = []
            for i in range(num_cams):
                intrinsics.append(parse_txt(intrinsics_files[i]))

                pose = parse_txt(pose_files[i])

                # convert from COLMAP/OpenCV to nerfstudio camera (OpenGL/Blender)
                pose[0:3, 1:3] *= -1

                camera_to_worlds.append(pose)

            intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))  # [N, 4, 4]
            camera_to_worlds = torch.from_numpy(np.stack(camera_to_worlds).astype(np.float32))  # [N, 4, 4]

            return intrinsics, camera_to_worlds, num_cams

        # get all split cam params
        intrinsics_train, camera_to_worlds_train, n_train = get_camera_params("train")
        intrinsics_val, camera_to_worlds_val, n_val = get_camera_params("val")
        intrinsics_test, camera_to_worlds_test, _ = get_camera_params("test")

        # combine all cam params
        intrinsics = torch.cat([intrinsics_train, intrinsics_val, intrinsics_test], dim=0)
        camera_to_worlds = torch.cat([camera_to_worlds_train, camera_to_worlds_val, camera_to_worlds_test], dim=0)

        # as all images are ~on-a-plane, align that plane with z-up
        if self.config.align_poses_method == "ransac":
            w2zup, _ = get_world_to_z_up(camera_to_worlds)
            camera_to_worlds = w2zup @ camera_to_worlds

            camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds, method="none", center_poses=self.config.center_poses
            )

            # Scale poses
            scale_factor = 1.0
            if self.config.auto_scale_poses:
                scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

            camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

            # add an offset to x and y so that the object are centered
            # TODO this is hard coded per scene, there must be a better way to do this?
            if self.scene == "stjacob":
                camera_to_worlds[:, 0, 3] += -0.3  # x
                camera_to_worlds[:, 1, 3] += -0.5  # y
            if self.scene == "europa":
                camera_to_worlds[:, 0, 3] += -0.1  # x
                camera_to_worlds[:, 1, 3] += -0.65  # y
            if self.scene == "lk2":
                camera_to_worlds[:, 0, 3] += -0.0  # x
                camera_to_worlds[:, 1, 3] += -0.65  # y
        else:
            camera_to_worlds, _ = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=self.config.align_poses_method,
                center_poses=self.config.center_poses,
            )

            # Scale poses
            scale_factor = 1.0
            if self.config.auto_scale_poses:
                scale_factor /= torch.max(torch.abs(camera_to_worlds[:, :3, 3]))

            camera_to_worlds[:, :3, 3] *= scale_factor * self.config.scale_factor

        if split == "train":
            camera_to_worlds = camera_to_worlds[:n_train]
            intrinsics = intrinsics[:n_train]
        elif split == "val":
            camera_to_worlds = camera_to_worlds[n_train : n_train + n_val]
            intrinsics = intrinsics[n_train : n_train + n_val]
        elif split == "test":
            camera_to_worlds = camera_to_worlds[n_train + n_val :]
            intrinsics = intrinsics[n_train + n_val :]

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            camera_type=CameraType.PERSPECTIVE,
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            ),
            radius=aabb_scale,
            near=0.01,
            far=3.5 * aabb_scale,
            collider_type="near_far",
        )

        split = "validation" if split == "val" else split
        split_dir = f"{self.data}/{self.scene}/final/{split}"

        # --- images ---
        image_filenames = find_files(f"{split_dir}/rgb", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        # --- semantics --- (must be generated/downloaded, not provided in original dataset)
        panoptic_classes = load_from_json(self.config.data / "cityscapes_classes.json")
        classes = panoptic_classes["classes"]
        colors = torch.tensor(panoptic_classes["colours"], dtype=torch.float32) / 255.0
        segmentation_filenames = find_files(f"{split_dir}/cityscapes_mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        # --- capture sessions ---
        # images that start with the same date are from the same session
        if self.config.use_session_data:
            capture_sessions = {"sessions": {}, "image_idx_to_session": []}
            for i, filename in enumerate(image_filenames):
                session = filename.split("_IMG")[0].split("/")[-1]
                if session not in capture_sessions["sessions"]:
                    capture_sessions["sessions"][session] = {"image_ids": []}
                capture_sessions["sessions"][session]["image_ids"].append(i)
                capture_sessions["image_idx_to_session"].append(session)

            # --- environment maps ---
            resize = Resize(
                (self.config.environment_map_resolution // 2, self.config.environment_map_resolution),
                InterpolationMode.NEAREST,
            )
            for session in capture_sessions["sessions"]:
                envmap_filename = find_files(
                    f"{self.scene_dir}/ENV_MAP_CC/{session}", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"]
                )
                semantic_filename = find_files(
                    f"{self.scene_dir}/ENV_MAP_CC/{session}/cityscapes_mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"]
                )

                # check if envmap and semantic segmentation are not available
                if len(envmap_filename) == 0:
                    capture_sessions["sessions"][session]["envmap"] = "none"
                    capture_sessions["sessions"][session]["envmap_semantics"] = "none"
                    capture_sessions["sessions"][session]["envmap_fg_mask"] = "none"
                else:
                    envmap = ToTensor()(Image.open(envmap_filename[0]))  # shape (3, H, W)
                    envmap = resize(envmap)  # shape (3, H, W)
                    envmap = envmap.permute(1, 2, 0)  # shape (H, W, 3)

                    segmentation_image = ToTensor()(Image.open(semantic_filename[0]))  # shape (3, H, W)
                    segmentation_image = resize(segmentation_image)  # shape (3, H, W)
                    segmentation_image = segmentation_image.permute(1, 2, 0)  # shape (H, W, 3)

                    envmap_fg_mask = torch.zeros_like(segmentation_image[:, :, 0])  # shape (H, W)

                    envmap_fg_mask = torch.where(
                        torch.all(torch.eq(segmentation_image, colors[classes.index("sky")]), dim=2),
                        torch.ones_like(envmap_fg_mask),
                        envmap_fg_mask,
                    )
                    capture_sessions["sessions"][session]["envmap"] = envmap
                    capture_sessions["sessions"][session]["envmap_semantics"] = segmentation_image
                    capture_sessions["sessions"][session]["envmap_fg_mask"] = envmap_fg_mask

        # --- masks ---
        mask_filenames = None
        masks = None
        fg_masks = None
        semantic_masks = None
        if self.config.masks_from_semantics and self.config.use_masks:
            masks = []
            fg_masks = []
            semantic_masks = []
            for semantic_filename in segmentation_filenames:
                segmentation_image = ToTensor()(Image.open(semantic_filename))  # shape (3, H, W)
                segmentation_image = segmentation_image.permute(1, 2, 0)  # shape (H, W, 3)
                # Create an empty mask with the same shape as your image

                mask = torch.ones_like(segmentation_image[:, :, 0])
                fg_mask = torch.ones_like(segmentation_image[:, :, 0])

                mask = torch.where(
                    torch.all(torch.eq(segmentation_image, colors[classes.index("person")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("rider")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("car")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("truck")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("bus")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("train")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("motorcycle")]), dim=2)
                    | torch.all(torch.eq(segmentation_image, colors[classes.index("bicycle")]), dim=2),
                    torch.zeros_like(mask),
                    mask,
                )

                fg_mask = torch.where(
                    torch.all(torch.eq(segmentation_image, colors[classes.index("sky")]), dim=2),
                    torch.zeros_like(fg_mask),
                    fg_mask,
                )

                masks.append(mask)
                fg_masks.append(fg_mask)
                semantic_masks.append(segmentation_image)
        else:
            # --- masks ---
            if self.config.use_masks is not None:
                mask_filenames = find_files(f"{split_dir}/mask", exts=["*.png", "*.jpg", "*.JPG", "*.PNG"])

        additional_inputs = {
            "masks": {
                "func": get_masks,
                "kwargs": {"masks": masks, "fg_masks": fg_masks, "semantic_masks": semantic_masks},
            }
        }
        if self.config.use_session_data:
            additional_inputs["capture_sessions"] = {
                "func": get_session_data,
                "kwargs": {"capture_sessions": capture_sessions},
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs,
        )

        return dataparser_outputs
