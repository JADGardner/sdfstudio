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
Base class for the graphs.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Type

import icosphere
import torch
from torch import nn

from nerfstudio.configs.base_config import InstantiateConfig


class IcosahedronSampler(nn.Module):
    """For sampling directions from an icosahedron."""

    def __init__(self, icosphere_order: int = 2):
        super().__init__()
        self.icosphere_order = icosphere_order

        vertices, _ = icosphere.icosphere(self.icosphere_order)
        self.directions = torch.from_numpy(vertices).float()  # [N, 3], # Z is up

    def forward(self, positions):
        """Returns directions for each position.

        Args:
            positions: [num_rays, samples_per_ray, 3]

        Returns:
            directions: [num_rays, samples_per_ray, num_directions, 3]
        """

        return self.directions[None, None, :, :].expand(positions.shape[0], positions.shape[1], -1, -1)


# Field related configs
@dataclass
class IlluminationFieldConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IlluminationField)
    """target class to instantiate"""


class IlluminationField(nn.Module):
    """Base class for illumination fields."""

    def __init__(
        self,
        sampling_method: Literal["icosphere", "other", "None"] = "icosphere",
        icosphere_order: int = 2,
    ) -> None:
        super().__init__()
        self.sampling_method = sampling_method
        self.icosphere_order = icosphere_order

        self.setup_sampler()

    def setup_sampler(self):
        """Initializes the sampler.

        Raises:
            NotImplementedError: Choosen sampling method is not implemented.
        """
        if self.sampling_method == "icosphere":
            self.sampler = IcosahedronSampler(self.icosphere_order)
        elif self.sampling_method == "None":
            self.sampler = None
        else:
            raise NotImplementedError

    def get_directions(self, positions):
        """Gets directions for each position."""

        return self.sampler(positions)

    @abstractmethod
    def get_outputs(self, unique_indices, inverse_indices, directions):
        """Computes and returns the colors. Returns output field values.

        Args:
            unique_indices: [rays_per_batch]
            inverse_indices: [rays_per_batch, samples_per_ray]
            directions: [rays_per_batch, samples_per_ray, num_directions, 3]
        """

    def forward(self, camera_indices, positions, directions=None):
        """Selects directions and evaluates the field for each camera.

        Args:
            camera_indicies: [rays_per_batch, samples_per_ray]
            positions: [rays_per_batch, samples_per_ray, 3]
            directions: [rays_per_batch, samples_per_ray, num_directions, 3]
        """
        if directions is None:
            directions = self.get_directions(positions)  # [rays_per_batch, samples_per_ray, num_directions, 3]
        unique_indices, inverse_indices = torch.unique(camera_indices, return_inverse=True)
        illumination_colours, illumination_directions = self.get_outputs(unique_indices, inverse_indices, directions)
        return illumination_colours, illumination_directions
