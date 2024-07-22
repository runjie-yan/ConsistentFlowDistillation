from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    BaseImplicitGeometryGenerator,
    contract_to_unisphere,
)
from threestudio.utils.typing import *


@threestudio.register("multi-particles-geometry")
class MultiPtcGeometry(BaseImplicitGeometryGenerator):
    @dataclass
    class Config(BaseImplicitGeometryGenerator.Config):
        particle_num: int = 4
        base_geometry_type: str = ""
        base_geometry: Optional[BaseGeometry.Config] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.geometry_list = [
            threestudio.find(self.cfg.base_geometry_type)(self.cfg.base_geometry).to(
                self.device
            )
            for _ in range(self.cfg.particle_num)
        ]
        self.encoding_list = nn.ModuleList([geo.encoding for geo in self.geometry_list])
        self.density_network_list = nn.ModuleList(
            [geo.density_network for geo in self.geometry_list]
        )
        self.feature_network_list = nn.ModuleList(
            [geo.feature_network for geo in self.geometry_list]
        )
        # select current particle id
        self.particle_id = 0

    def regenerate(self, gen_id: Optional[int] = None):
        if gen_id is None:
            self.particle_id = np.random.randint(0, self.cfg.particle_num)
        else:
            self.particle_id = gen_id

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ):
        return self.geometry_list[self.particle_id].get_activated_density(
            points, density
        )

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        return self.geometry_list[self.particle_id].forward(points, output_normal)

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        return self.geometry_list[self.particle_id].forward_density(points)

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        return self.geometry_list[self.particle_id].forward_field(points)

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return self.geometry_list[self.particle_id].forward_level(field, threshold)

    # TODO: export ?
    # TODO: create from ?
