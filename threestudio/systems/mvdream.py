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


@threestudio.register("mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        rgb_as_latents: bool = False
        enable_eval_metirc: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        self.freq_timer = freq_timer()
        if self.cfg.enable_eval_metirc:
            from torchmetrics.multimodal.clip_score import CLIPScore

            # to keep on cpu
            self.metrics = {
                "CLIP_B16": CLIPScore(
                    model_name_or_path="openai/clip-vit-base-patch16"
                ),
                "CLIP_B32": CLIPScore(
                    model_name_or_path="openai/clip-vit-base-patch32"
                ),
                "CLIP_L14": CLIPScore(
                    model_name_or_path="openai/clip-vit-large-patch14"
                ),
                "CLIP_L14_336": CLIPScore(
                    model_name_or_path="openai/clip-vit-large-patch14-336"
                ),
            }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        noise_out = self.noise_generator(out, batch)
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        guidance_out = self.guidance(
            out["comp_rgb"],
            self.prompt_utils,
            **batch,
            **schedule_out,
            **noise_out,
            rgb_as_latents=False,
        )

        loss = 0.0

        # log and losses
        self.log("training_speed", self.freq_timer.get_freq())
        self.log("train_params/noise_sqrt_beta", self.noise_generator.sqrt_beta)
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        if self.cfg.enable_eval_metirc:
            for mtc_name, metirc in self.metrics.items():
                self.metrics[mtc_name] = metirc.to(self.device)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        noise_out = self.noise_generator(out, batch)
        noise_vis = F.interpolate(
            noise_out["noise"][:, :3],
            (512, 512),
            mode="nearest",
        )
        # metric evaluation
        if self.cfg.enable_eval_metirc:
            with torch.no_grad():
                prompt = self.prompt_utils.prompt
                clip_img = (out["comp_rgb"].permute(0, 3, 1, 2)[0] * 255).int()
                for mtc_name, metirc in self.metrics.items():
                    self.log(
                        f"metric/{mtc_name}", metirc(clip_img, prompt), reduce_fx="max"
                    )

        if not self.cfg.rgb_as_latents:
            with torch.no_grad():
                latents = self.guidance.encode_images(
                    out["comp_rgb"].permute(0, 3, 1, 2)
                )
                img_enc_dec = self.guidance.decode_latents(latents).permute(0, 2, 3, 1)
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
                        "img": img_enc_dec[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if not self.cfg.rgb_as_latents
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": noise_vis[0].permute(1, 2, 0),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                }
            ]
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
                        "img": out["z_mean"][0, :, :, 0],
                        "kwargs": {"data_range": (0.5, 1.5)},
                    },
                ]
                if "z_mean" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        if self.cfg.enable_eval_metirc:
            for mtc_name, metirc in self.metrics.items():
                self.metrics[mtc_name] = metirc.cpu()

    def on_test_epoch_start(self) -> None:
        if self.cfg.enable_eval_metirc:
            for mtc_name, metirc in self.metrics.items():
                self.metrics[mtc_name] = metirc.to(self.device)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        noise_out = self.noise_generator(out, batch)
        noise_vis = F.interpolate(
            noise_out["noise"][:, :3],
            (512, 512),
            mode="nearest",
        )
        # metric evaluation
        if self.cfg.enable_eval_metirc:
            with torch.no_grad():
                prompt = self.prompt_utils.prompt
                clip_img = (out["comp_rgb"].permute(0, 3, 1, 2)[0] * 255).int()
                for mtc_name, metirc in self.metrics.items():
                    self.log(
                        f"metric-test/{mtc_name}",
                        metirc(clip_img, prompt),
                        reduce_fx="max",
                    )

        if not self.cfg.rgb_as_latents:
            with torch.no_grad():
                img_enc_dec = self.guidance.decode_latents(
                    self.guidance.encode_images(out["comp_rgb"].permute(0, 3, 1, 2))
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
                        "img": img_enc_dec[0],
                        # "kwargs": {"data_format": "HWC", "data_range": (-0.1, 0.1)},
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if not self.cfg.rgb_as_latents
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": noise_vis[0].permute(1, 2, 0),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                }
            ]
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
                        "img": out["z_mean"][0, :, :, 0],
                        "kwargs": {"data_range": (0.5, 1.5)},
                    },
                ]
                if "z_mean" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        if self.cfg.enable_eval_metirc:
            for mtc_name, metirc in self.metrics.items():
                self.metrics[mtc_name] = metirc.cpu()
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
