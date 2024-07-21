import math

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


class ProgressiveBandFrequency(nn.Module, Updateable):
    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        global_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            threestudio.debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class TCNNEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, x):
        return self.encoding(x)


# 4D implicit decomposition of space and time (4D-fy)
class TCNNEncodingSpatialTime(nn.Module):
    def __init__(
        self, in_channels, config, dtype=torch.float32, init_time_zero=False
    ) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        config["otype"] = "HashGrid"
        self.num_frames = 1  # config["num_frames"]
        self.static = config["static"]
        self.cfg = config_to_primitive(config)
        self.cfg_time = self.cfg
        self.n_key_frames = config.get("n_key_frames", 1)
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(self.n_input_dims, self.cfg, dtype=dtype)
            self.encoding_time = tcnn.Encoding(
                self.n_input_dims + 1, self.cfg_time, dtype=dtype
            )
        self.n_output_dims = self.encoding.n_output_dims
        self.frame_time = None
        if self.static:
            self.set_temp_param_grad(requires_grad=False)
        self.use_key_frame = config.get("use_key_frame", False)
        self.is_video = True
        self.update_occ_grid = False

    def set_temp_param_grad(self, requires_grad=False):
        self.set_param_grad(self.encoding_time, requires_grad=requires_grad)

    def set_param_grad(self, param_list, requires_grad=False):
        if isinstance(param_list, nn.Parameter):
            param_list.requires_grad = requires_grad
        else:
            for param in param_list.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        # TODO frame_time only supports batch_size == 1 cases
        if self.update_occ_grid and not isinstance(self.frame_time, float):
            frame_time = self.frame_time
        else:
            if (self.static or not self.training) and self.frame_time is None:
                frame_time = torch.zeros(
                    (self.num_frames, 1), device=x.device, dtype=x.dtype
                ).expand(x.shape[0], 1)
            else:
                if self.frame_time is None:
                    frame_time = 0.0
                else:
                    frame_time = self.frame_time
                frame_time = (
                    torch.ones((self.num_frames, 1), device=x.device, dtype=x.dtype)
                    * frame_time
                ).expand(x.shape[0], 1)
            frame_time = frame_time.view(-1, 1)
        enc_space = self.encoding(x)
        x_frame_time = torch.cat((x, frame_time), 1)
        enc_space_time = self.encoding_time(x_frame_time)
        enc = enc_space + enc_space_time
        return enc


class ProgressiveBandHashGrid(nn.Module, Updateable):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "Grid"
        encoding_config["type"] = "Hash"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step, on_load_weights=False):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            threestudio.debug(f"Update current level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0

class TriplaneEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        assert in_channels == 3, f"{type(self).__name__} does not support input channel that != 3"
        self.init_mode: str = config['init_mode']
        self.tensorf: bool = config['tensorf']
        self.op_resolution: str = config['op_resolution']

        self.n_features_per_level: int = config['n_features_per_level']
        self.base_resolution: int = config['base_resolution']
        self.per_level_scale: float = config['per_level_scale']
        self.n_levels: int = config['n_levels']
        self.with_parameters: bool = config['with_parameters']

        self.resolutions = [
            int(self.base_resolution * self.per_level_scale ** i)
            for i in range(self.n_levels) 
        ]
        if self.with_parameters:
            self.triplanes = nn.ParameterList([
                nn.Parameter(
                    torch.empty((3, self.n_features_per_level, res, res), dtype=dtype)
                )
                for res in self.resolutions
            ])
        else:
            self.register_parameter('triplanes', None)
        
        if self.tensorf and self.with_parameters:
            self.trilines = nn.ParameterList([
                nn.Parameter(
                    torch.empty((3, self.n_features_per_level, res, 1), dtype=dtype)
                )
                for res in self.resolutions
            ])
        else:
            self.register_parameter('trilines', None)
        self.reset_parameters()
        
        self.n_input_dims = in_channels
        if self.op_resolution == 'cat':
            self.n_output_dims = 3*self.n_features_per_level*self.n_levels
        elif self.op_resolution == 'mean':
            self.n_output_dims = 3*self.n_features_per_level
        
    def reset_parameters(self) -> None:
        if self.with_parameters:
            if self.init_mode == 'normal':
                for triplane in self.triplanes:
                    nn.init.normal_(triplane)
                if self.tensorf:
                    for triline in self.trilines:
                        nn.init.normal_(triline)
            elif self.init_mode == 'zeros':
                for triplane in self.triplanes:
                    nn.init.zeros_(self.triplanes)
                if self.tensorf:
                    for triline in self.trilines:
                        nn.init.zeros_(triline)
        
    def _grid_sample(self, points: Float[Tensor, "3 *N Di"], grid: Float[Tensor, "S Df G1 G2"]) -> Float[Tensor, "*N SxDf"]:
        points_shape = points.shape[1:-1]
        df = self.n_features_per_level
        di = 2
        out = F.grid_sample(
            grid.float(), points.view(3, 1, -1, di).float(), align_corners=False, mode="bilinear"
        )
        out = out.reshape(df*3, -1).T.reshape(*points_shape, df*3)
        return out
        
    def forward(
        self, 
        x: Float[Tensor, "*N 3"], 
        triplanes: Optional[List[Float[Tensor, "3 C R R"]]] = None,
        trilines: Optional[List[Float[Tensor, "3 C R 1"]]] = None,
    ):
        # precomputes
        if triplanes is None:
            triplanes = self.triplanes 
        if trilines is None:
            trilines = self.trilines 
        x = x * 2 - 1  # convert to [-1, 1] for grid sample
        x_pl = torch.stack([
                x[...,[0, 2]],
                x[...,[1, 0]],
                x[...,[2, 1]],
            ],
            dim=0            
        )
        if self.tensorf:
            x_li = torch.stack([
                    x[...,0],
                    x[...,1],
                    x[...,2],
                ],
                dim=0            
            )
            x_li = torch.stack([x_li, torch.zeros_like(x_li)], dim=-1)
            
        feat_level = []
        for res_id in range(self.n_levels):
            x_feat = self._grid_sample(x_pl, triplanes[res_id])
            if self.tensorf:
                x_feat = x_feat * self._grid_sample(x_li, trilines[res_id])
            feat_level.append(x_feat)
        if self.op_resolution == 'cat':
            feat_level = torch.cat(feat_level, dim=-1)
        elif self.op_resolution == 'mean':
            feat_level = torch.stack(feat_level, dim=-1).mean(dim=-1)
        return feat_level
        

class CompositeEncoding(nn.Module, Updateable):
    def __init__(self, encoding, include_xyz=False, xyz_scale=2.0, xyz_offset=-1.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args, **kwargs):
        return (
            self.encoding(x, *args, **kwargs)
            if not self.include_xyz
            else torch.cat(
                [x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args, **kwargs)], dim=-1
            )
        )


def get_encoding(n_input_dims: int, config) -> nn.Module:
    # input suppose to be range [0, 1]
    encoding: nn.Module
    if config.otype == "ProgressiveBandFrequency":
        encoding = ProgressiveBandFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    elif config.otype == "HashGridSpatialTime":
        encoding = TCNNEncodingSpatialTime(n_input_dims, config)  # 4D-fy encoding
    elif config.otype == "TriplaneEncoding":
        encoding = TriplaneEncoding(n_input_dims, config)
    else:
        encoding = TCNNEncoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding

class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, n_power_iterations=1):
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())
        self.n_power_iterations = n_power_iterations
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        self._update_u()
        return nn.functional.linear(input, self.weight * self.u, self.bias)

    def _update_u(self):
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = torch.nn.functional.normalize(torch.matmul(self.u, self.weight.data), dim=1)
                u = torch.nn.functional.normalize(torch.matmul(v, self.weight.data.transpose(0, 1)), dim=1)
            self.u.copy_(u)

class SpectralMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = SpectralNormLinear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)
    
class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)
    
class VanillaCNN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_conv(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_conv(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_conv(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x: Float[Tensor, "B H W C"]):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = x.permute(0,3,1,2)
            x = self.layers(x)
            x = self.output_activation(x)
            x = x.permute(0,2,3,1)
        return x

    def make_conv(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class SphereInitVanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = True, True
        self.sphere_init_radius = config["sphere_init_radius"]
        self.sphere_init_inside_out = config["inside_out"]

        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)

        if is_last:
            if not self.sphere_init_inside_out:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            else:
                torch.nn.init.constant_(layer.bias, self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=-math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
        elif is_first:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
            torch.nn.init.normal_(
                layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out)
            )
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        return nn.Softplus(beta=100)


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)

    def forward(self, x):
        return self.network(x).float()  # transform to float32


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    network: nn.Module
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SpectralMLP":
        network = SpectralMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "VanillaCNN":
        network = VanillaCNN(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SphereInitVanillaMLP":
        network = SphereInitVanillaMLP(
            n_input_dims, n_output_dims, config_to_primitive(config)
        )
    else:
        assert (
            config.get("sphere_init", False) is False
        ), "sphere_init=True only supported by VanillaMLP"
        network = TCNNNetwork(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))


class TCNNNetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
    ) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network_with_input_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def forward(self, x):
        return self.network_with_input_encoding(x).float()  # transform to float32


def create_network_with_input_encoding(
    n_input_dims: int, n_output_dims: int, encoding_config, network_config
) -> nn.Module:
    # input suppose to be range [0, 1]
    network_with_input_encoding: nn.Module
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP", "SphereInitVanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        network_with_input_encoding = NetworkWithInputEncoding(encoding, network)
    else:
        network_with_input_encoding = TCNNNetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=config_to_primitive(encoding_config),
            network_config=config_to_primitive(network_config),
        )
    return network_with_input_encoding


class ToDTypeWrapper(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)
