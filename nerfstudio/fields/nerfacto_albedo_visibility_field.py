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


from typing import Dict, Optional, Tuple

import icosphere
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.colormaps import sRGB

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0
    # return directions


class TCNNRENINerfactoAlbedoField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        num_layers_normal: int = 3,
        hidden_dim_normal: int = 64,
        output_dim_normal: int = 64,
        num_layers_visibility: int = 3,
        hidden_dim_visibility: int = 64,
        output_dim_visibility: int = 64,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        num_layers_semantics: int = 2,
        hidden_dim_semantics: int = 64,
        output_dim_semantics: int = 64,
        use_pred_normals: bool = True,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_visibility: bool = True,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.use_visibility = use_visibility

        base_res = 16
        features_per_level = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # semantics
        if self.use_semantics:
            self.mlp_semantics = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=output_dim_semantics,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_semantics,
                    "n_hidden_layers": num_layers_semantics - 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        # explicit visibility prediction
        if self.use_visibility:
            self.mlp_visibility = tcnn.Network(
                n_input_dims=self.position_encoding.n_output_dims
                + self.direction_encoding.n_output_dims
                + self.geo_feat_dim,
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

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.position_encoding.n_output_dims,
                n_output_dims=output_dim_normal,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_normal,
                    "n_hidden_layers": num_layers_normal - 1,
                },
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

        # colour network
        self.mlp_head = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        if torch.isnan(positions).any():
            raise ValueError("positions has nan")

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)

        if torch.isnan(h).any():
            # save output of positions_flat and h
            raise ValueError("h has nan")

        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # check density and base_mlp_out for nan
        if torch.isnan(density_before_activation).any():
            torch.save(density_before_activation, "density_before_activation.pt")
            torch.save(base_mlp_out, "base_mlp_out.pt")
            raise ValueError("density_before_activation has nan")
        if torch.isnan(base_mlp_out).any():
            raise ValueError("base_mlp_out has nan")

        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None, illumination_directions=None
    ):
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        # camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(
            ray_samples.frustums.directions
        )  # [num_rays, samples_per_ray, 3] # all samples have same direction

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # semantics
        if self.use_semantics:
            density_embedding_copy = density_embedding.clone().detach()
            semantics_input = torch.cat(
                [
                    density_embedding_copy.view(-1, self.geo_feat_dim),
                ],
                dim=-1,
            )
            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            normals = self.field_head_pred_normals(x)
            outputs[FieldHeadNames.PRED_NORMALS] = normals

        if torch.isnan(density_embedding).any():
            raise ValueError("NaN in density embedding.")

        albedo = self.mlp_head(density_embedding.view(-1, self.geo_feat_dim)).view(*outputs_shape, -1).to(directions)

        albedo = albedo.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update({FieldHeadNames.ALBEDO: albedo})

        # check if nan in rgb
        if torch.isnan(albedo).any():
            raise ValueError("NaN in rgb, prior to lambertian shading.")

        illumination_visibility = None
        if self.use_visibility:
            if self.spatial_distortion is not None:
                positions = ray_samples.frustums.get_positions()  # [num_rays, samples_per_ray, 3]
                positions = self.spatial_distortion(positions)  # [num_rays, samples_per_ray, 3]
                positions = (positions + 2.0) / 4.0  # [num_rays, samples_per_ray, 3]
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            positions_flat = self.position_encoding(positions.view(-1, 3))  # [num_rays * samples_per_ray, 16]
            directions = illumination_directions  # [D, 3]
            directions = self.direction_encoding(directions)  # [D, 16] (SH encoding, order 4)
            directions = directions.unsqueeze(0).repeat(positions_flat.shape[0], 1, 1)  # [K, D, 16]
            density_emb = density_embedding.view(-1, self.geo_feat_dim)  # [K, 15]
            density_emb = density_emb.unsqueeze(1).repeat(1, directions.shape[1], 1)  # [K, D, 15]
            positions_flat = positions_flat.unsqueeze(1).repeat(1, directions.shape[1], 1)  # [K, D, 3]
            directions = directions.view(-1, directions.shape[-1])  # [K * D, 16]
            density_emb = density_emb.view(-1, density_emb.shape[-1])  # [K * D, 15]
            positions_flat = positions_flat.view(-1, positions_flat.shape[-1])  # [K * D, 16]
            visibility_input = torch.cat([positions_flat, directions, density_emb], dim=-1)
            x = self.mlp_visibility(visibility_input)  # [K * D, output_dim_visibility]
            illumination_visibility = self.field_head_visibility(x.float())
            illumination_visibility = illumination_visibility.view(
                -1, illumination_directions.shape[0], 1
            ).squeeze()  # [K, D]

            # now for camera rays to apply loss from sky masks
            if self.spatial_distortion is not None:
                positions = ray_samples.frustums.get_positions()  # [num_rays, samples_per_ray, 3]
                positions = self.spatial_distortion(positions)  # [num_rays, samples_per_ray, 3]
                positions = (positions + 2.0) / 4.0  # [num_rays, samples_per_ray, 3]
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            positions_flat = self.position_encoding(positions.view(-1, 3))  # [num_rays, samples_per_ray, 16] -> [K, 16]
            directions = self.direction_encoding(
                ray_samples.frustums.directions.reshape(-1, 3)
            )  # [K, 16] (SH encoding, order 4)
            density_emb = density_embedding.view(-1, self.geo_feat_dim)  # [K, 15]
            visibility_input = torch.cat([positions_flat, directions, density_emb], dim=-1)
            x = self.mlp_visibility(visibility_input)  # [K * D, output_dim_visibility]
            visibility = self.field_head_visibility(x.float())
            visibility = visibility.view(positions.shape[0], positions.shape[1], 1)  # [K, D, 1]
            outputs.update({FieldHeadNames.VISIBILITY: visibility})

        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, illumination_directions=None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding, illumination_directions=illumination_directions
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


class TorchRENINerfactoAlbedoField(Field):
    """
    PyTorch implementation of the compound field.
    """

    def __init__(
        self,
        aabb,
        num_images: int,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        skip_connections: Tuple = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
        else:
            positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(
                torch.cat(
                    [
                        encoded_dir,
                        density_embedding,  # type:ignore,
                    ],
                    dim=-1,  # type:ignore
                )
            )
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs


field_implementation_to_class = {"tcnn": TCNNRENINerfactoAlbedoField, "torch": TorchRENINerfactoAlbedoField}
