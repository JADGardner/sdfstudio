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

"""Classic NeRF field"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


def so3_invariant_representation(Z, D, conditioning: Optional[str] = "Concat"):
    """Generates an SO3 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)
        conditioning (str): Type of conditioning to use. Options are 'Concat' and 'FiLM'

    Returns:
        torch.Tensor: SO3 invariant representation (B x npix x ndims + ndims^2)
    """
    G = Z @ torch.transpose(Z, 1, 2)
    innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
    z_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    if conditioning == "FiLM":
        return innerprod, z_invar
    if conditioning == "Concat":
        return torch.cat((innerprod, z_invar), 2)
    raise ValueError(f"Invalid conditioning type {conditioning}")


def so2_invariant_representation(Z, D, conditioning: Optional[str] = "Concat"):
    """Generates an SO2 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)

    Returns:
        torch.Tensor: SO2 invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    d_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
    G = torch.bmm(z_xz, torch.transpose(z_xz, 1, 2))
    # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
    z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    # innerprod is size B x npix x ndims
    innerprod = torch.bmm(d_xz, torch.transpose(z_xz, 1, 2))
    d_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
    # Copy Z_y for every pixel to be size B x npix x ndims
    z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
    # Just the y component of D (B x npix x 1)
    d_y = D[:, :, 1].unsqueeze(2)
    if conditioning == "FiLM":
        model_input = torch.cat((d_xz_norm, d_y, innerprod), 2)  # [B, npix, 2 + ndims]
        conditioning_input = torch.cat((z_xz_invar, z_y), 2)  # [B, npix, ndims^2 + ndims]
        return model_input, conditioning_input
    if conditioning == "Concat":
        # model_input is size B x npix x 2 x ndims + ndims^2 + 2
        model_input = torch.cat((innerprod, z_xz_invar, d_xz_norm, z_y, d_y), 2)
        return model_input
    raise ValueError(f"Invalid conditioning type {conditioning}")


def no_invariance(Z, D, conditioning: Optional[str] = "Concat"):
    """Generates an representation from latent code Z and direction coordinates D
       that is not invariant to rotation.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)

    Returns:
        torch.Tensor: Latent representation that is not invariant (ndims * 3 + ndims)
    """
    innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
    z_input = Z.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    if conditioning == "FiLM":
        return innerprod, z_input
    if conditioning == "Concat":
        model_input = torch.cat((innerprod, z_input), 2)
        return model_input
    raise ValueError(f"Invalid conditioning type {conditioning}")


class UnMinMaxNormlise(object):
    def __init__(self, minmax):
        self.minmax = minmax

    def __call__(self, x):
        x = 0.5 * (x + 1) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
        x = torch.exp(x)
        return x


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class RENIField(Field):
    """RENI Field

    Args:
        dataset_size (int): Number of images in dataset
        ndims (int): Number of latent dimensions
        equivariance (str): Type of equivariance to use. Options are 'SO3', 'SO2', and 'None'
        hidden_features (int): Number of features in hidden layers
        hidden_layers (int): Number of hidden layers
        out_features (int): Number of features in output layer
        last_layer_linear (bool): Whether to use a linear layer in the last layer
        output_activation (str): Type of activation to use in output layer. Options are 'ReLU', 'Sigmoid', and 'None'
        first_omega_0 (float): Initial value of first layer's omega_0
        hidden_omega_0 (float): Initial value of hidden layers' omega_0
        minmax (tuple): Values to use in minmax normalization
        fixed_decoder (bool): Whether to use a fixed decoder
    """

    def __init__(
        self,
        dataset_size,
        ndims,
        equivariance,
        hidden_features,
        hidden_layers,
        out_features,
        last_layer_linear,
        output_activation,
        first_omega_0,
        hidden_omega_0,
        minmax,
        fixed_decoder,
    ):
        super().__init__()
        # set all hyperaparameters from config
        self.dataset_size = dataset_size
        self.ndims = ndims
        self.equivariance = equivariance
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.last_layer_linear = last_layer_linear
        self.output_activation = output_activation
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.minmax = minmax
        self.fixed_decoder = fixed_decoder

        if self.equivariance == "None":
            self.invariant_representation = no_invariance
            self.in_features = self.ndims * 3 + self.ndims
        elif self.equivariance == "SO2":
            self.invariant_representation = so2_invariant_representation
            self.in_features = 2 * self.ndims + self.ndims * self.ndims + 2
        elif self.equivariance == "SO3":
            self.invariant_representation = so3_invariant_representation
            self.in_features = self.ndims + self.ndims * self.ndims

        self.init_latent_codes(self.dataset_size, self.ndims, self.fixed_decoder)

        self.unnormalise = UnMinMaxNormlise(self.minmax)

        self.net = []

        self.net.append(
            SineLayer(
                self.in_features,
                self.hidden_features,
                is_first=True,
                omega_0=self.first_omega_0,
            )
        )

        for _ in range(self.hidden_layers):
            self.net.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.last_layer_linear:
            final_linear = nn.Linear(self.hidden_features, self.out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                    np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    self.hidden_features,
                    self.out_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.output_activation == "exp":
            self.net.append(torch.exp)
        elif self.output_activation == "tanh":
            self.net.append(nn.Tanh())

        self.net = nn.Sequential(*self.net)

        if self.fixed_decoder:
            for param in self.net.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v

        if self.fixed_decoder:
            net_state_dict = {}
            for key in new_state_dict.keys():
                if key.startswith("net."):
                    net_state_dict[key[4:]] = new_state_dict[key]
            self.net.load_state_dict(net_state_dict, strict=strict)
        else:
            super().load_state_dict(new_state_dict, strict=strict)

    def get_density(self, ray_samples: RaySamples):
        return None, None

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
