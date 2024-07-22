from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometryGenerator
from threestudio.systems.base import BaseDistill2DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.timer import freq_timer
from threestudio.utils.typing import *


@threestudio.register("distillation-2d-system")
class Distill2DSystem(BaseDistill2DSystem):
    @dataclass
    class Config(BaseDistill2DSystem.Config):
        rgb_as_latents: bool = False
        enable_snr_metric: bool = False

    cfg: Config

    def configure(self):
        # create imgenerator, noise_generator, t_scheduler
        super().configure()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_utils = self.prompt_processor()
        self.freq_timer = freq_timer()

    def forward(self, batch: Dict[str, Any], vis=False) -> Dict[str, Any]:
        render_out = self.imgenerator(**batch)
        if vis and self.cfg.rgb_as_latents:
            render_out["comp_rgb"] = self.guidance.vae_decode(
                self.guidance.pipe.vae, render_out["comp_rgb"].permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
        return {
            **render_out,
        }

    def training_step(self, batch, batch_idx):
        self.imgenerator.regenerate()
        out = self(batch)
        noise_out = self.noise_generator(out, batch)
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        guidance_out = self.guidance(
            out["comp_rgb"],
            self.prompt_utils,
            **batch,
            **schedule_out,
            **noise_out,
            rgb_as_latents=self.cfg.rgb_as_latents,
        )

        loss = 0.0

        # log and losses
        if self.cfg.enable_snr_metric:
            optimizer = self.optimizers()
            for group in optimizer.param_groups:
                if group["name"] == "imgenerator.embedding":
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
                        # print(snr_grad)
                    else:
                        # optimizer not initialized
                        pass

        self.log("training_speed", self.freq_timer.get_freq())
        self.log("train_params/noise_sqrt_beta", self.noise_generator.sqrt_beta)
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        return {"loss": loss}

    def on_validation_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        self.imgenerator.regenerate()  # always regenerate for 2d
        out = self(batch, vis=True)
        noise_out = self.noise_generator(out, batch)
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        guidance_inp = out["comp_rgb"]
        guidance_out = self.guidance(
            out["comp_rgb"],
            self.prompt_utils,
            **batch,
            **schedule_out,
            **noise_out,
            rgb_as_latents=False,
            return_rgb_1step_orig=True,
        )

        noise_vis = F.interpolate(
            noise_out["noise"][:, :3],
            (512, 512),
            mode="nearest",
        )
        det_mask_vis = F.interpolate(
            noise_out["det_mask"].float(),
            (512, 512),
            mode="nearest",
        )
        if not self.cfg.rgb_as_latents:
            with torch.no_grad():
                img_enc_dec = self.guidance.vae_decode(
                    self.guidance.pipe.vae, guidance_out["latents"]
                ).permute(0, 2, 3, 1)
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
                        "img": img_enc_dec[0] - out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (-0.1, 0.1)},
                    },
                ]
                if not self.cfg.rgb_as_latents
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": guidance_out["rgb_1step_orig"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "rgb_1step_orig" in guidance_out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": guidance_out["rgb_1step_orig_phi"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "rgb_1step_orig_phi" in guidance_out
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": noise_vis.permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                }
            ]
            + [
                {
                    "type": "grayscale",
                    "img": det_mask_vis.permute(0, 2, 3, 1)[0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0.0, 1.0)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
            # texts=(["rgb"] if "comp_rgb" in out else [])
            # + (["x_decode-rgb"] if not self.cfg.rgb_as_latents else [])
            # + (["x_gt"] if "rgb_1step_orig" in guidance_out else [])
            # + (["x_vsd"] if "rgb_1step_orig_phi" in guidance_out else [])
            # + ["noise"] + ["noise_det_mask"]
        )

    def test_step(self, batch, batch_idx):
        self.imgenerator.regenerate()  # always regenerate for 2d
        out = self(batch, vis=True)
        noise_out = self.noise_generator(out, batch)
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        try:
            # if --test only
            guidance_out = self.guidance(
                out["comp_rgb"],
                self.prompt_utils,
                **batch,
                **schedule_out,
                **noise_out,
                rgb_as_latents=False,
                return_rgb_1step_orig=True,
            )
        except AttributeError:
            guidance_out = {}

        noise_vis = F.interpolate(
            noise_out["noise"][:, :3],
            (512, 512),
            mode="nearest",
        )
        det_mask_vis = F.interpolate(
            noise_out["det_mask"].float(),
            (512, 512),
            mode="nearest",
        )
        if not self.cfg.rgb_as_latents:
            with torch.no_grad():
                img_enc_dec = self.guidance.vae_decode(
                    self.guidance.pipe.vae, guidance_out["latents"]
                ).permute(0, 2, 3, 1)
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
                        "img": img_enc_dec[0] - out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (-0.1, 0.1)},
                    },
                ]
                if not self.cfg.rgb_as_latents
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": guidance_out["rgb_1step_orig"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "rgb_1step_orig" in guidance_out
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": noise_vis.permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                }
            ]
            + [
                {
                    "type": "grayscale",
                    "img": det_mask_vis.permute(0, 2, 3, 1)[0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0.0, 1.0)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
            # texts=(["rgb"] if "comp_rgb" in out else [])
            # + (["x_decode-rgb"] if not self.cfg.rgb_as_latents else [])
            # + (["x_gt"] if "rgb_1step_orig" in guidance_out else [])
            # + ["noise"] + ["noise_det_mask"]
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
