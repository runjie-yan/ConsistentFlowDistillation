from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometryGenerator
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.timer import freq_timer
from threestudio.utils.typing import *


@threestudio.register("reconstruction-system")
class ReconstructionSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        rgb_as_latents: bool = False
        enable_snr_metric: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.freq_timer = freq_timer()
        if self.cfg.enable_snr_metric:
            self.automatic_optimization = False
            self.snr_optimizer = optim.Adam(
                [
                    {
                        "params": self.geometry.encoding.parameters(),
                        "name": "geometry.encoding",
                    }
                ],
                lr=0.0,
                betas=[0.99, 0.99],
            )

    def forward(self, batch: Dict[str, Any], vis=False) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        if vis and self.cfg.rgb_as_latents:
            render_out["comp_rgb"] = self.guidance.decode_latents(
                render_out["comp_rgb"].permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        # set identity
        if isinstance(self.geometry, BaseImplicitGeometryGenerator):
            self.geometry.regenerate()
        out = self(batch)

        loss = 0.0

        # log and losses
        self.log("training_speed", self.freq_timer.get_freq())
        loss_fn = nn.MSELoss(
            reduction="sum"
        )  # there are some strange precision problems
        loss_mse = loss_fn(out["comp_rgb"], batch["rgb_gt"]) * self.C(
            self.cfg.loss.lambda_mse
        )
        self.log(
            "train/loss_mse",
            loss_mse,
            prog_bar=True,
            logger=True,
        )
        loss += loss_mse

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.cfg.enable_snr_metric:
            optimizer = self.snr_optimizer
            opt = self.optimizers()

            for group in optimizer.param_groups:
                if group["name"] == "geometry.encoding":
                    beta1, beta2 = group["betas"]
                    p = group["params"][0]
                    state = optimizer.state[p]
                    if "step" in state:
                        step = state["step"]
                        exp_avg_norm = state["exp_avg"] / (1 - beta1**step)
                        exp_avg_sq_norm = state["exp_avg_sq"] / (1 - beta2**step)
                        snr_grad = (exp_avg_norm**2).sum() / (
                            1e-8 + exp_avg_sq_norm.sum()
                        )
                        self.log("train/grad_snr", snr_grad)
                    else:
                        # optimizer not initialized
                        pass
            optimizer.zero_grad()
            opt.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            opt.step()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if isinstance(self.geometry, BaseImplicitGeometryGenerator):
            self.geometry.regenerate()
        out = self(batch, vis=True)
        self.save_image_grid(
            (
                f"train/it{self.true_global_step}-{batch['index'][0]}.png"
                if not batch["index"][0] == 0
                else f"train-video/{self.true_global_step}.png"
            ),
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb_gt"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0, :, :, 0],
                        "kwargs": {"data_range": (0.5, 1.5)},
                    },
                ]
                if "depth" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # if isinstance(self.geometry, BaseImplicitGeometryGenerator):
        #     self.geometry.regenerate()
        out = self(batch, vis=True)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb_gt"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0, :, :, 0],
                        "kwargs": {"data_range": (0.5, 1.5)},
                    },
                ]
                if "depth" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        try:
            self.save_img_sequence(
                f"it{self.true_global_step}-train",
                f"train-video",
                "(\d+)\.png",
                save_format="mp4",
                fps=10,
                name="train",
                step=self.true_global_step,
            )
        except:
            threestudio.warn("can not generate training video")
