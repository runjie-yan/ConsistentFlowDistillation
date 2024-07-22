from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class BaseImgenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # rgb_as_latents: bool = False
        n_output_dims: int = 3
        height: Any = 64
        width: Any = 64

    cfg: Config

    def regenerate(self, gen_id=None) -> None:
        raise NotImplementedError

    def forward(self, **batch) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError
