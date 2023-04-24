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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import icosphere
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHeadNames, FieldHead
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.sdf_field import SDFFieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFAlbedoVisibilityFieldConfig(SDFFieldConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SDFAlbedoVisibilityField)
    icosphere_order: int = 3
    use_visibility: Literal["none", "mlp", "proposal_network"] = "none"
    num_layers_visibility: int = 3
    hidden_dim_visibility: int = 64
    output_dim_visibility: int = 64
    """Number of light sample directions per point"""


class SDFAlbedoVisibilityField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SDFAlbedoVisibilityFieldConfig

    def __init__(
        self,
        config: SDFAlbedoVisibilityFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor
        self.icosphere_order = self.config.icosphere_order
        self.use_visibility = self.config.use_visibility
        num_layers_visibility = self.config.num_layers_visibility
        hidden_dim_visibility = self.config.hidden_dim_visibility
        output_dim_visibility = self.config.output_dim_visibility

        num_levels = 16
        max_res = 2048
        base_res = 16
        log2_hashmap_size = 19
        features_per_level = 2
        use_hash = True
        smoothstep = True
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )
        elif self.config.encoding_type == "periodic":
            print("using periodic encoding")
            self.encoding = PeriodicVolumeEncoding(
                num_levels=num_levels,
                min_res=base_res,
                max_res=max_res,
                log2_hashmap_size=18,  # 64 ** 3 = 2^18
                features_per_level=features_per_level,
                smoothstep=smoothstep,
            )
        elif self.config.encoding_type == "tensorf_vm":
            print("using tensor vm")
            self.encoding = TensorVMEncoding(128, 24, smoothstep=smoothstep)

        # TODO make this configurable
        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, include_input=False
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.geometric_init:
                if l == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "glin" + str(l), lin)

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)

        # color network (albedo)
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        # feature
        in_dim = 3 + self.config.geo_feat_dim
        # in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims + self.config.geo_feat_dim
        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)

        # explicit illumination visibility prediction
        if self.use_visibility == "mlp":
            self.mlp_visibility = tcnn.Network(
                n_input_dims=3
                + self.position_encoding.get_out_dim()
                + self.encoding.n_output_dims
                + self.direction_encoding.get_out_dim(),
                # + self.config.geo_feat_dim,
                n_output_dims=output_dim_visibility,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_visibility,
                    "n_hidden_layers": num_layers_visibility - 1,
                },
            )
            self.field_head_visibility = DensityFieldHead(
                in_dim=self.mlp_visibility.n_output_dims, activation=torch.sigmoid
            )
            self.field_head_termination = FieldHead(
                field_head_name=FieldHeadNames.TERMINATION,
                in_dim=self.mlp_visibility.n_output_dims,
                out_dim=1,
                activation=torch.sigmoid,
            )

            if self.use_visibility not in ["mlp"]:
                self.mlp_visibility.requires_grad_(False)
                self.field_head_visibility.requires_grad_(False)
                self.field_head_termination.requires_grad_(False)

        vertices, _ = icosphere.icosphere(self.icosphere_order)
        self.icosphere = torch.from_numpy(vertices).float()

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.exp = torch.exp

        self._cos_anneal_ratio = 1.0

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.use_grid_feature:
            # TODO check how we should normalize the points
            # normalize point range as encoding assume points are in [-1, 1]
            # positions = inputs / self.divide_factor
            positions = self.spatial_distortion(inputs)

            positions = (positions + 1.0) / 2.0
            feature = self.encoding(positions)
            # raise NotImplementedError
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def forward_visibility(self, ray_samples: RaySamples):
        """forward the visibility network"""
        origins = ray_samples.frustums.origins  # [num_rays * num_directions, 3]
        origins = origins.view(-1, 3)  # [K, 3]
        positions = self.spatial_distortion(origins)
        positions = (positions + 1.0) / 2.0
        feature = self.encoding(positions)
        pe = self.position_encoding(origins)
        positions_flat = torch.cat((origins, pe, feature), dim=-1)  # [K, N]

        directions = ray_samples.frustums.directions  # [num_rays * num_directions, 3] aka [K, 3]
        directions = directions.view(-1, 3)  # [K, 3]
        directions = self.direction_encoding(directions)  # [K, 16] (SH encoding, order 4)

        # final input to the visibility network
        visibility_input = torch.cat([positions_flat, directions], dim=-1)

        x = self.mlp_visibility(visibility_input)  # [K, output_dim_visibility]
        visibility = self.field_head_visibility(x.float())  # [K, 1]
        termination_dist = self.field_head_termination(x.float())
        return visibility, termination_dist

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def get_sdf_at_pos(self, positions):
        """predict the sdf value for ray samples"""
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat)
        sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def gradient(self, x):
        """compute the gradient of the ray"""
        x.requires_grad_(True)
        y = self.forward_geonetwork(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return gradients

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs = inputs.view(-1, 3)  # [num_rays * samples_per_ray, 3]
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_albedo(self, positions_flat, geo_features):
        """compute diffuse albedo"""
        # positions = self.spatial_distortion(positions_flat)
        # positions = (positions + 1.0) / 2.0  # needed for SH encoding
        # feature = self.encoding(positions)
        # pe = self.position_encoding(positions_flat)
        # h_pos = torch.cat((positions_flat, pe, feature), dim=-1)
        # h = torch.cat([h_pos, geo_features.view(-1, self.config.geo_feat_dim)], dim=-1)

        h = torch.cat([positions_flat, geo_features.view(-1, self.config.geo_feat_dim)], dim=-1)
        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        albedo = self.sigmoid(h)  # [num_rays * samples_per_ray, 3]
        return albedo

    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas=False,
        return_occupancy=False,
        illumination_directions=None,
    ):  # pylint: disable=arguments-renamed
        """compute output of ray samples"""

        outputs = {}

        positions = ray_samples.frustums.get_start_positions()  # [num_rays, samples_per_ray, 3]
        positions_flat = positions.view(-1, 3)  # [num_rays * samples_per_ray, 3]

        directions = ray_samples.frustums.directions  # [num_rays, samples_per_ray, 3] # all samples have same direction

        positions_flat.requires_grad_(True)

        with torch.enable_grad():
            h = self.forward_geonetwork(positions_flat)
            sdf, geo_features = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)

        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=positions_flat,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        density = self.laplace_density(sdf)

        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        albedo = self.get_albedo(positions_flat, geo_features)

        # sky_visibility_sample = None
        # if self.use_visibility == "mlp":
        #     inputs = ray_samples.frustums.get_start_positions()
        #     inputs_flat = inputs.view(-1, 3)  # [num_rays * samples_per_ray, 3] aka [K, 3]
        #     positions = self.spatial_distortion(inputs_flat)
        #     positions = (positions + 1.0) / 2.0
        #     feature = self.encoding(positions)
        #     pe = self.position_encoding(inputs_flat)
        #     positions_flat = torch.cat((inputs_flat, pe, feature), dim=-1)
        #     # directions = self.icosphere.to(positions.device)  # [D, 3]
        #     directions = illumination_directions  # [D, 3]
        #     directions = self.direction_encoding(directions)  # [D, 16] (SH encoding, order 4)
        #     directions = directions.unsqueeze(0).repeat(positions_flat.shape[0], 1, 1)  # [K, D, 16]
        #     density_emb = geo_features.view(-1, self.config.geo_feat_dim)  # [K, 15]
        #     density_emb = density_emb.unsqueeze(1).repeat(1, directions.shape[1], 1)  # [K, D, 15]
        #     positions_flat = positions_flat.unsqueeze(1).repeat(1, directions.shape[1], 1)  # [K, D, 3]
        #     directions = directions.view(-1, directions.shape[-1])  # [K * D, 16]
        #     density_emb = density_emb.view(-1, density_emb.shape[-1])  # [K * D, 15]
        #     positions_flat = positions_flat.view(-1, positions_flat.shape[-1])  # [K * D, 16]
        #     visibility_input = torch.cat([positions_flat, directions, density_emb], dim=-1)
        #     x = self.mlp_visibility(visibility_input)  # [K * D, output_dim_visibility]
        #     sky_visibility_sample = self.field_head_visibility(x.float())
        #     sky_visibility_sample = sky_visibility_sample.view(
        #         -1, sky_visibility_sample.shape[0], 1
        #     ).squeeze()  # [K, D] aka [num_rays * samples_per_ray, num_illumination_directions]
        #     sky_termination_sample = self.field_head_termination(x.float())
        #     sky_termination_sample = sky_termination_sample.view(
        #         -1, sky_termination_sample.shape[0], 1
        #     )  # [K, D, 1] aka [num_rays * samples_per_ray, num_illumination_directions, 3]

        #     # now for camera rays to apply loss from sky masks
        #     inputs = ray_samples.frustums.get_start_positions()
        #     inputs_flat = inputs.view(-1, 3)  # [num_rays * samples_per_ray, 3]
        #     positions = self.spatial_distortion(inputs_flat)
        #     positions = (positions + 1.0) / 2.0
        #     feature = self.encoding(positions)
        #     pe = self.position_encoding(inputs_flat)
        #     positions_flat = torch.cat((inputs_flat, pe, feature), dim=-1)

        #     directions = self.direction_encoding(
        #         ray_samples.frustums.directions.reshape(-1, 3)
        #     )  # [K, 16] (SH encoding, order 4)
        #     density_emb = geo_features.view(-1, self.config.geo_feat_dim)  # [K, 15]
        #     visibility_input = torch.cat([positions_flat, directions, density_emb], dim=-1)
        #     x = self.mlp_visibility(visibility_input)  # [K * D, output_dim_visibility]
        #     sky_visibility_camera_ray = self.field_head_visibility(x.float())
        #     sky_visibility_camera_ray = sky_visibility_camera_ray.view(inputs.shape[0], inputs.shape[1], 1)  # [K, D, 1]
        #     sky_termination_camera_ray = self.field_head_termination(x.float())
        #     sky_termination_camera_ray = sky_termination_camera_ray.view(
        #         inputs.shape[0], inputs.shape[1], 1
        #     )  # [K, D, 1]

        albedo = albedo.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: albedo,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        # if self.use_visibility == "mlp":
        #     outputs.update({"sky_visibility_sample": sky_visibility_sample})
        #     outputs.update({"sky_termination_sample": sky_termination_sample})
        #     outputs.update({"sky_visibility_camera_ray": sky_visibility_camera_ray})
        #     outputs.update({"sky_termination_camera_ray": sky_termination_camera_ray})

        return outputs

    def forward(
        self,
        ray_samples: RaySamples,
        return_alphas=False,
        return_occupancy=False,
        illumination_directions=None,
    ):  # pylint: disable=arguments-renamed
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(
            ray_samples,
            return_alphas=return_alphas,
            return_occupancy=return_occupancy,
            illumination_directions=illumination_directions,
        )
        return field_outputs
