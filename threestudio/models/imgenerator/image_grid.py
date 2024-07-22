from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.imgenerator.base import BaseImgenerator
from threestudio.models.networks import get_mlp
from threestudio.utils.typing import *


@threestudio.register("image-grid")
class ImageGrid(BaseImgenerator):
    @dataclass
    class Config(BaseImgenerator.Config):
        embedding_frozen: bool = False
        n_hidden_dims: int = 32
        mlp_decoding: bool = False
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        init_mode: str = "normal"

    cfg: Config

    def configure(self):
        super().configure()
        if self.cfg.mlp_decoding:
            self.decoder = get_mlp(
                self.cfg.n_hidden_dims,
                self.cfg.n_output_dims,
                self.cfg.mlp_network_config,
            )
        else:
            if self.cfg.n_hidden_dims != self.cfg.n_output_dims:
                threestudio.warn(
                    "n_hidden_dims!=n_output_dims and not mlp decoding, ignore n_output_dims"
                )
            self.register_module("decoder", None)

        self.embedding = nn.Parameter(
            torch.empty((1, self.cfg.height, self.cfg.width, self.cfg.n_hidden_dims))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.cfg.init_mode == "normal":
            nn.init.normal_(self.embedding)
        elif self.cfg.init_mode == "zeros":
            nn.init.zeros_(self.embedding)

    def regenerate(self, gen_id=None) -> None:
        pass

    def forward(self, **batch) -> Dict[str, Float[Tensor, "..."]]:
        B = batch["batch_size"]
        embedding = self.embedding.repeat(B, 1, 1, 1)
        if self.cfg.mlp_decoding:
            embedding = self.decoder(embedding)
        return {"comp_rgb": embedding}
