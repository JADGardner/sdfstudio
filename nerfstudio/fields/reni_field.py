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
Implementation of Rotation Equivariant Natural Illumination (RENI).
"""

from functools import singledispatch, update_wrapper

import numpy as np
import torch
from torch import nn

from nerfstudio.fields.base_illumination_field import (
    IlluminationField,
    IlluminationFieldConfig,
)

####################################################################
###################### ↓↓↓↓↓ UTILS ↓↓↓↓↓ ###########################
####################################################################


def methdispatch(func):
    """
    A decorator that allows the defining of a function (or method)
    with multiple dispatch based on the type of one of its arguments.

    The decorator uses the `singledispatch` function from the `functools`
    module to create the dispatcher. The new function will call the
    original function, but with a different implementation based on
    the type of the second argument (`args[1]`). The `register`
    attribute of the new function allows you to define additional
    implementations for the original function based on the type
    of the second argument.

    Args:
        func (callable): The function to be decorated. This should be a
        function that takes at least two arguments.

    Returns:
        callable: A new function that acts as a dispatcher for the
        original function. The new function will call the original
        function with a different implementation based on the type
        of the second argument.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def get_directions(sidelen):
    """Generates a flattened grid of (x,y,z,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2,sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen,2]
    theta = np.pi * (uv[:, 0] - 1)
    phi = np.pi * uv[:, 1]
    directions = torch.stack(
        (
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
            -torch.sin(phi) * torch.cos(theta),
        ),
        -1,
    ).unsqueeze(
        0
    )  # shape=[1, sidelen/2*sidelen, 3]
    return directions


# sine of the polar angle for compensation of irregular equirectangular sampling
def get_sineweight(sidelen):
    """Returns a matrix of sampling densites"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2, sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen, 2]
    phi = np.pi * uv[:, 1]
    sineweight = torch.sin(phi)  # [sidelen/2*sidelen]
    sineweight = sineweight.unsqueeze(1).repeat(1, 3).unsqueeze(0)  # shape=[1, sidelen/2*sidelen, 3]
    return sineweight


####################################################################
########## ↓↓↓↓↓ CONDITIONING VIA CONCATENATION ↓↓↓↓↓ ##############
####################################################################


def so3_invariant_representation(Z, D):
    """Generates an SO3 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)

    Returns:
        torch.Tensor: SO3 invariant representation (B x npix x ndims + ndims^2)
    """
    G = Z @ torch.transpose(Z, 1, 2)
    innerprod = torch.bmm(D, torch.transpose(Z, 1, 2))
    Z_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    model_input = torch.cat((innerprod, Z_invar), 2)
    return model_input


def so2_invariant_representation(Z, D):
    """Generates an SO2 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x npix x 3)

    Returns:
        torch.Tensor: SO2 invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    D_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
    G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
    # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
    Z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    # innerprod is size B x npix x ndims
    innerprod = torch.bmm(D_xz, torch.transpose(Z_xz, 1, 2))
    D_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
    # Copy Z_y for every pixel to be size B x npix x ndims
    Z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
    # Just the y component of D (B x npix x 1)
    D_y = D[:, :, 1].unsqueeze(2)
    # Conditioning via concatenation
    model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
    # model_input is size B x npix x 2 x ndims + ndims^2 + 2
    return model_input


def so2_invariant_representation_4D(Z, D):
    """Generates an SO2 invariant representation from latent code Z and direction coordinates D.

    Args:
        Z (torch.Tensor): Latent code (B x N x ndims x 3)
        D (torch.Tensor): Direction coordinates (B x N x 3)

    Returns:
        torch.Tensor: SO2 invariant representation (B x npix x 2 x ndims + ndims^2 + 2)
    """
    Z = Z.reshape(-1, Z.shape[-2], Z.shape[-1])  # (K x ndims x 3)
    D = D.reshape(-1, D.shape[-1])  # (K x 3)
    z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    d_xz = torch.stack((D[:, 0], D[:, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size npix x ndims x ndims
    G = torch.bmm(z_xz, torch.transpose(z_xz, 1, 2))
    z_xz_invar = G.flatten(start_dim=1).unsqueeze(0)  # (1 x K x ndims^2)
    # innerprod is size B x npix x ndims unsqueeze to add batch dim
    innerprod = torch.einsum("ij,ikj->ik", d_xz, z_xz).unsqueeze(0)  # (1 x K x ndims)
    d_xz_norm = torch.norm(d_xz, dim=1).unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
    z_y = Z[:, :, 1].unsqueeze(0)  # (1 x K x ndims)
    # Just the y component of D
    d_y = D[:, 1].unsqueeze(0).unsqueeze(2)  # (1 x K x 1)
    # Conditioning via concatenation
    model_input = torch.cat((innerprod, z_xz_invar, d_xz_norm, z_y, d_y), 2)
    # model_input is size B x npix x 2 x ndims + ndims^2 + 2
    return model_input


def no_invariance(Z, D):
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
    model_input = torch.cat((innerprod, z_input), 2)
    return model_input


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


class RENIAutoDecoder(nn.Module):
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
        fixed_decoder,
    ):
        super().__init__()
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
        self.fixed_decoder = fixed_decoder

        if self.equivariance == "None":
            self.InvariantRepresentation = no_invariance
            self.in_features = self.ndims * 3 + self.ndims
        elif self.equivariance == "SO2":
            self.InvariantRepresentation = so2_invariant_representation
            self.InvariantRepresentation4D = so2_invariant_representation_4D
            self.in_features = 2 * self.ndims + self.ndims * self.ndims + 2
        elif self.equivariance == "SO3":
            self.InvariantRepresentation = so3_invariant_representation
            self.in_features = self.ndims + self.ndims * self.ndims

        self.init_latent_codes(self.dataset_size, self.ndims, fixed_decoder=fixed_decoder)

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

    def sample_latent(self, idx):
        return self.Z[idx, :, :], None, None

    def init_latent_codes(self, dataset_size, ndims, fixed_decoder=False):
        if fixed_decoder:
            self.Z = nn.Parameter(torch.zeros(dataset_size, ndims, 3))
        else:
            self.Z = nn.Parameter(torch.randn((dataset_size, ndims, 3)))

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

    @methdispatch
    def forward(self, x, directions):
        raise NotImplementedError(
            "x must be either an int (idx), torch.Tensor (idxs or latent codes) or a list of ints (idxs)"
        )

    @forward.register
    def _(self, idx: int, directions):
        assert len([idx]) == directions.shape[0]
        Z = self.Z[[idx], :, :]
        x = self.InvariantRepresentation(Z, directions)
        return self.net(x)

    @forward.register
    def _(self, idx: list, directions):
        assert len(idx) == directions.shape[0]
        Z = self.Z[idx, :, :]
        x = self.InvariantRepresentation(Z, directions)
        return self.net(x)

    @forward.register
    def _(self, x: torch.Tensor, directions):
        if len(x.shape) == 1:
            idx = x
            Z = self.Z[idx, :, :]
            x = self.InvariantRepresentation(Z, directions)
        elif len(x.shape) == 4:
            Z = x
            x = self.InvariantRepresentation4D(Z, directions)
        else:
            Z = x
            x = self.InvariantRepresentation(Z, directions)
        return self.net(x)


class RENIVariationalAutoDecoder(nn.Module):
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
            self.InvariantRepresentation = no_invariance
            self.in_features = self.ndims * 3 + self.ndims
        elif self.equivariance == "SO2":
            self.InvariantRepresentation = so2_invariant_representation
            self.InvariantRepresentation4D = so2_invariant_representation_4D
            self.in_features = 2 * self.ndims + self.ndims * self.ndims + 2
        elif self.equivariance == "SO3":
            self.InvariantRepresentation = so3_invariant_representation
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

    def get_Z(self):
        return self.mu

    def sample_latent(self, idx):
        """Sample the latent code at a given index

        Args:
        idx (int): Index of the latent variable to sample

        Returns:
        tuple (torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the sampled latent variable, the mean of the latent variable and the log variance of the latent variable
        """
        if torch.is_grad_enabled():
            mu = self.mu[idx, :, :]
            log_var = self.log_var[idx, :, :]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
            return sample, mu, log_var
        else:
            return self.mu[idx, :, :], self.mu[idx, :, :], self.log_var[idx, :, :]

    def init_latent_codes(self, dataset_size, ndims, fixed_decoder=True):
        self.log_var = torch.nn.Parameter(torch.normal(-5, 1, size=(dataset_size, ndims, 3)))
        if fixed_decoder:
            self.mu = nn.Parameter(torch.zeros(dataset_size, ndims, 3))
            self.log_var.requires_grad = False
        else:
            self.mu = nn.Parameter(torch.randn((dataset_size, ndims, 3)))

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

    @methdispatch
    def forward(self, x, directions):
        raise NotImplementedError(
            "x must be either an int (idx), torch.Tensor (idxs or latent codes) or a list of ints (idxs)"
        )

    @forward.register
    def _(self, idx: int, directions):
        assert len([idx]) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[[idx], :, :]
        else:
            Z, _, _ = self.sample_latent([idx])
        x = self.InvariantRepresentation(Z, directions)
        return self.net(x)

    @forward.register
    def _(self, idx: list, directions):
        assert len(idx) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[idx, :, :]
        else:
            Z, _, _ = self.sample_latent(idx)
        x = self.InvariantRepresentation(Z, directions)
        return self.net(x)

    @forward.register
    def _(self, x: torch.Tensor, directions):
        if len(x.shape) == 1:
            idx = x
            if self.fixed_decoder:
                Z = self.mu[idx, :, :]
            else:
                Z, _, _ = self.sample_latent(idx)
            x = self.InvariantRepresentation(Z, directions)
        elif len(x.shape) == 4:
            Z = x
            x = self.InvariantRepresentation4D(Z, directions)
        else:
            Z = x
            x = self.InvariantRepresentation(Z, directions)
        return self.net(x)


####################################################################
################# ↓↓↓↓↓ FiLM CONDITIONING ↓↓↓↓↓ ####################
####################################################################


def SO3InvariantRepresentationFiLM(Z, D):
    # Invariant representation of Z, gram matrix G=Z*Z'
    G = Z @ torch.transpose(Z, 1, 2)  # [B, ndims, ndims]
    Siren_Input = torch.bmm(D, torch.transpose(Z, 1, 2))  # [B, npix, ndims]
    # Flatten G and replicate for all pixels
    Mapping_Input = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)  # [B, npix, ndims^2]
    return Siren_Input, Mapping_Input


def SO2InvariantRepresentationFiLM(Z, D):
    # Derives a representation from latent code Z and direction coordinates D that is
    # invariant under y-axis rotation of Z and D simultaneously
    # B = Batchsize
    # Z is B x ndims x 3
    # D is B x npix x 3
    Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    D_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z'
    G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))  # [B, ndims, ndims]
    # Flatten G and replicate for all pixels
    Z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)  # [B, npix, ndims^2]
    innerprod = torch.bmm(D_xz, torch.transpose(Z_xz, 1, 2))  # [B, npix, ndims]
    D_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)  # [B, npix, 1]
    # Copy Z_y for every pixel
    Z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)  # [B, npix, ndims]
    # Just the y component of D
    D_y = D[:, :, 1].unsqueeze(2)  # [B, npix, 1]
    # Conditioning via FiLM
    Siren_Input = torch.cat((D_xz_norm, D_y, innerprod), 2)  # [B, npix, 2 + ndims]
    Mapping_Input = torch.cat((Z_xz_invar, Z_y), 2)  # [B, npix, ndims^2 + ndims]
    return Siren_Input, Mapping_Input


def NoInvarianceFiLM(Z, D):
    Siren_Input = torch.bmm(D, torch.transpose(Z, 1, 2))
    Mapping_Input = Z.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    return Siren_Input, Mapping_Input


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class CustomMappingNetwork(nn.Module):
    def __init__(self, in_features, map_hidden_layers, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = []

        for _ in range(map_hidden_layers):
            self.network.append(nn.Linear(in_features, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = map_hidden_dim

        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))

        self.network = nn.Sequential(*self.network)

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor")]
        phase_shifts = frequencies_offsets[..., torch.div(frequencies_offsets.shape[-1], 2, rounding_mode="floor") :]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return torch.sin(freq * x + phase_shift)


class RENIAutoDecoderFiLM(nn.Module):
    def __init__(
        self,
        dataset_size,
        ndims,
        equivariance,
        siren_hidden_features,
        siren_hidden_layers,
        mapping_network_features,
        mapping_network_layers,
        out_features,
        output_activation,
        fixed_decoder,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.ndims = ndims
        self.equivariance = equivariance
        self.siren_hidden_features = siren_hidden_features
        self.siren_hidden_layers = siren_hidden_layers
        self.mapping_network_features = mapping_network_features
        self.mapping_network_layers = mapping_network_layers
        self.out_features = out_features
        self.output_activation = output_activation
        self.fixed_decoder = fixed_decoder

        if self.equivariance == "None":
            self.InvariantRepresentation = NoInvarianceFiLM
            self.in_features = self.ndims * 3
            self.mn_in_features = self.ndims
        elif self.equivariance == "SO2":
            self.InvariantRepresentation = SO2InvariantRepresentationFiLM
            self.in_features = 2 + self.ndims
            self.mn_in_features = self.ndims * self.ndims + self.ndims
        elif self.equivariance == "SO3":
            self.InvariantRepresentation = SO3InvariantRepresentationFiLM
            self.in_features = self.ndims
            self.mn_in_features = self.ndims * self.ndims

        self.init_latent_codes(self.dataset_size, self.ndims, self.fixed_decoder)

        self.net = nn.ModuleList()

        self.net.append(FiLMLayer(self.in_features, self.siren_hidden_features))

        for _ in range(self.siren_hidden_layers - 1):
            self.net.append(FiLMLayer(self.siren_hidden_features, self.siren_hidden_features))

        self.final_layer = nn.Linear(self.siren_hidden_features, self.out_features)

        self.mapping_network = CustomMappingNetwork(
            self.mn_in_features,
            self.mapping_network_layers,
            self.mapping_network_features,
            (len(self.net)) * self.siren_hidden_features * 2,
        )

        self.net.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.net[0].apply(first_layer_film_sine_init)

        if self.output_activation == "exp":
            self.final_activation = torch.exp
        elif self.output_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        if self.fixed_decoder:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False
            for param in self.mapping_network.parameters():
                param.requires_grad = False

    def init_latent_codes(self, dataset_size, ndims, fixed_decoder=False):
        if fixed_decoder:
            self.Z = nn.Parameter(torch.zeros(dataset_size, ndims, 3))
        else:
            self.Z = torch.nn.Parameter(torch.randn((dataset_size, ndims, 3)))

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v

        if self.fixed_decoder:
            net_state_dict = {}
            mapping_network_state_dict = {}
            for key in new_state_dict.keys():
                if key.startswith("net."):
                    net_state_dict[key[4:]] = new_state_dict[key]
                elif key.startswith("mapping_network."):
                    mapping_network_state_dict[key[16:]] = new_state_dict[key]
            self.net.load_state_dict(net_state_dict, strict=strict)
            self.mapping_network.load_state_dict(mapping_network_state_dict, strict=strict)
            self.final_layer.weight = nn.Parameter(new_state_dict["final_layer.weight"], requires_grad=False)
            self.final_layer.bias = nn.Parameter(new_state_dict["final_layer.bias"], requires_grad=False)
        else:
            super().load_state_dict(new_state_dict, strict=strict)

    @methdispatch
    def forward(self, x, directions):
        raise NotImplementedError(
            "x must be either an int (idx), torch.Tensor (idxs or latent codes) or a list of ints (idxs)"
        )

    @forward.register
    def _(self, idx: int, directions):
        assert len([idx]) == directions.shape[0]
        Z = self.Z[[idx], :, :]
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    @forward.register
    def _(self, idx: list, directions):
        assert len(idx) == directions.shape[0]
        Z = self.Z[idx, :, :]
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    @forward.register
    def _(self, x: torch.Tensor, directions):
        if len(x.shape) == 1:
            idx = x
            Z = self.Z[idx, :, :]
        else:
            Z = x
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, x, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30

        for index, layer in enumerate(self.net):
            start = index * self.siren_hidden_features
            end = (index + 1) * self.siren_hidden_features
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        x = self.final_layer(x)
        output = self.final_activation(x)
        return output


class RENIVariationalAutoDecoderFiLM(nn.Module):
    def __init__(
        self,
        dataset_size,
        ndims,
        equivariance,
        siren_hidden_features,
        siren_hidden_layers,
        mapping_network_features,
        mapping_network_layers,
        out_features,
        output_activation,
        fixed_decoder,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.ndims = ndims
        self.equivariance = equivariance
        self.siren_hidden_features = siren_hidden_features
        self.siren_hidden_layers = siren_hidden_layers
        self.mapping_network_features = mapping_network_features
        self.mapping_network_layers = mapping_network_layers
        self.out_features = out_features
        self.output_activation = output_activation
        self.fixed_decoder = fixed_decoder

        if self.equivariance == "None":
            self.InvariantRepresentation = NoInvarianceFiLM
            self.in_features = self.ndims * 3
            self.mn_in_features = self.ndims
        elif self.equivariance == "SO2":
            self.InvariantRepresentation = SO2InvariantRepresentationFiLM
            self.in_features = 2 + self.ndims
            self.mn_in_features = self.ndims * self.ndims + self.ndims
        elif self.equivariance == "SO3":
            self.InvariantRepresentation = SO3InvariantRepresentationFiLM
            self.in_features = self.ndims
            self.mn_in_features = self.ndims * self.ndims

        self.init_latent_codes(self.dataset_size, self.ndims, self.fixed_decoder)

        self.net = nn.ModuleList()

        self.net.append(FiLMLayer(self.in_features, self.siren_hidden_features))

        for _ in range(self.siren_hidden_layers - 1):
            self.net.append(FiLMLayer(self.siren_hidden_features, self.siren_hidden_features))

        self.final_layer = nn.Linear(self.siren_hidden_features, self.out_features)

        self.mapping_network = CustomMappingNetwork(
            self.mn_in_features,
            self.mapping_network_layers,
            self.mapping_network_features,
            (len(self.net)) * self.siren_hidden_features * 2,
        )

        self.net.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.net[0].apply(first_layer_film_sine_init)

        if self.output_activation == "exp":
            self.final_activation = torch.exp
        elif self.output_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        if self.fixed_decoder:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False
            for param in self.mapping_network.parameters():
                param.requires_grad = False

    def sample_latent(self, idx):
        mu = self.mu[idx, :, :]
        log_var = self.log_var[idx, :, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample, mu, log_var

    def init_latent_codes(self, dataset_size, ndims, fixed_decoder=True):
        self.log_var = torch.nn.Parameter(torch.normal(-5, 1, size=(dataset_size, ndims, 3)))
        if fixed_decoder:
            self.mu = nn.Parameter(torch.zeros(dataset_size, ndims, 3))
            self.log_var.requires_grad = False
        else:
            self.mu = torch.nn.Parameter(torch.randn((dataset_size, ndims, 3)))

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v

        if self.fixed_decoder:
            net_state_dict = {}
            mapping_network_state_dict = {}
            for key in new_state_dict.keys():
                if key.startswith("net."):
                    net_state_dict[key[4:]] = new_state_dict[key]
                elif key.startswith("mapping_network."):
                    mapping_network_state_dict[key[16:]] = new_state_dict[key]
            self.net.load_state_dict(net_state_dict, strict=strict)
            self.mapping_network.load_state_dict(mapping_network_state_dict, strict=strict)
            self.final_layer.weight = nn.Parameter(new_state_dict["final_layer.weight"], requires_grad=False)
            self.final_layer.bias = nn.Parameter(new_state_dict["final_layer.bias"], requires_grad=False)
        else:
            super().load_state_dict(new_state_dict, strict=strict)

    @methdispatch
    def forward(self, x, directions):
        raise NotImplementedError(
            "x must be either an int (idx), torch.Tensor (idxs or latent codes) or a list of ints (idxs)"
        )

    @forward.register
    def _(self, idx: int, directions):
        assert len([idx]) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[[idx], :, :]
        else:
            Z, _, _ = self.sample_latent([idx])
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    @forward.register
    def _(self, idx: list, directions):
        assert len(idx) == directions.shape[0]
        if self.fixed_decoder:
            Z = self.mu[idx, :, :]
        else:
            Z, _, _ = self.sample_latent(idx)
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    @forward.register
    def _(self, x: torch.Tensor, directions):
        if len(x.shape) == 1:
            idx = x
            if self.fixed_decoder:
                Z = self.mu[idx, :, :]
            else:
                Z, _, _ = self.sample_latent(idx)
        else:
            Z = x
        Siren_Input, Mapping_Network_Input = self.InvariantRepresentation(Z, directions)
        frequencies, phase_shifts = self.mapping_network(Mapping_Network_Input)
        return self.forward_with_frequencies_phase_shifts(Siren_Input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, x, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30

        for index, layer in enumerate(self.net):
            start = index * self.siren_hidden_features
            end = (index + 1) * self.siren_hidden_features
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        x = self.final_layer(x)
        output = self.final_activation(x)
        return output


def get_reni_field(chkpt_path, num_latent_codes, fixed_decoder=True):
    """Returns a RENI model from a PyTorch Lightning checkpoint

    Args:
        chkpt_path (str): path to PyTorch Lightning checkpoint
        num_latent_codes (int): number of latent codes to use

    Returns:
        nn.Module: The initalised RENI model
    """
    chkpt = torch.load(chkpt_path)
    config = chkpt["hyper_parameters"]["config"]

    ### MODEL ###
    conditioning = config.RENI.CONDITIONING
    model_type = config.RENI.MODEL_TYPE
    equivariance = config.RENI.EQUIVARIANCE
    latent_dim = config.RENI.LATENT_DIMENSION
    hidden_layers = config.RENI.HIDDEN_LAYERS
    hidden_features = config.RENI.HIDDEN_FEATURES
    out_features = config.RENI.OUT_FEATURES
    last_layer_linear = config.RENI.LAST_LAYER_LINEAR
    output_activation = config.RENI.OUTPUT_ACTIVATION
    first_omega_0 = config.RENI.FIRST_OMEGA_0
    hidden_omega_0 = config.RENI.HIDDEN_OMEGA_0

    # model trained on minmax normalised log(HDR) images where minmax is a function
    # of the training dataset. We need to unnormalise the output of the model
    # for use in downstream tasks.
    minmax = [-18.0536, 11.4633]  # this is the minmax of the training dataset just taken from the config

    # create a model with the same hyperparameters as the checkpoint
    # but setting 'dataset_size' to the required number of latent
    # codes for your downstream task and 'fixed_decoder' to True
    if conditioning == "Cond-by-Concat":
        if model_type == "VariationalAutoDecoder":
            model = RENIVariationalAutoDecoder(
                num_latent_codes,
                latent_dim,
                equivariance,
                hidden_features,
                hidden_layers,
                out_features,
                last_layer_linear,
                output_activation,
                first_omega_0,
                hidden_omega_0,
                minmax,
                fixed_decoder=fixed_decoder,
            )

    model.load_state_dict(chkpt["state_dict"])
    return model
