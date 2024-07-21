import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.models.geometry.base import BaseImplicitGeometryGenerator
from threestudio.utils.timer import freq_timer

@threestudio.register("prolificdreamer-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        enable_eval_metirc: bool = False
        normal_as_rgb_prob: Any = 0.

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.strict_loading = False # FIXME
        self.freq_timer = freq_timer()
        if self.cfg.enable_eval_metirc:
            from torchmetrics.multimodal.clip_score import CLIPScore
            # to keep on cpu
            self.metrics = {
                "CLIP_B16": CLIPScore(model_name_or_path="openai/clip-vit-base-patch16"),
                "CLIP_B32": CLIPScore(model_name_or_path="openai/clip-vit-base-patch32"),
                "CLIP_L14": CLIPScore(model_name_or_path="openai/clip-vit-large-patch14"),
                "CLIP_L14_336": CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336"),
            }
            
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.normal_as_rgb_prob = self.C(self.cfg.normal_as_rgb_prob)
    
    def forward(self, batch: Dict[str, Any], vis=False) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        if not vis and random.random() < self.normal_as_rgb_prob:
            if 'comp_normal' in render_out:
                render_out['comp_rgb'] = render_out['comp_normal'] 
            else:
                threestudio.warn('do not found normal in render output')
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
        noise_out = self.noise_generator(out, batch)
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]
        schedule_out = self.t_scheduler(guidance_inp.shape[0])
        guidance_out = self.guidance(
            guidance_inp, 
            self.prompt_utils, 
            **batch, 
            **schedule_out, 
            **noise_out, 
            rgb_as_latents=False, 
        )

        loss = 0.0
            
        self.log("training_speed", self.freq_timer.get_freq())
        self.log("train_params/noise_sqrt_beta", self.noise_generator.sqrt_beta)
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
            
        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if "z_variance" in out:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            
        if self.cfg.stage == "coarse":
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

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)


            # sdf loss
            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        if loss.isnan():
            threestudio.warn("loss is NaN, stop running")
            exit(1)
                  
        return {"loss": loss}
    
    def on_validation_epoch_start(self) -> None:
        if self.cfg.enable_eval_metirc:
            for mtc_name, metirc in self.metrics.items():
                self.metrics[mtc_name] = metirc.to(self.device)

    def validation_step(self, batch, batch_idx):
        if isinstance(self.geometry, BaseImplicitGeometryGenerator):
            self.geometry.regenerate()
        out = self(batch, vis=True)
        noise_out = self.noise_generator(out, batch)
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]
        schedule_out = self.t_scheduler(guidance_inp.shape[0])
        guidance_out = self.guidance(
            guidance_inp, 
            self.prompt_utils, 
            **batch, 
            **schedule_out,
            **noise_out,
            rgb_as_latents=False, 
            return_rgb_1step_orig=True,
        )
        # metric evaluation
        if self.cfg.enable_eval_metirc:
            with torch.no_grad():
                prompt = self.prompt_utils.prompt
                clip_img = (out["comp_rgb"].permute(0,3,1,2)[0]*255).int()
                for mtc_name, metirc in self.metrics.items():
                    self.log(f"metric/{mtc_name}", metirc(clip_img, prompt), reduce_fx="max")
        
        self.save_image_grid(
            f"train/it{self.true_global_step}-{batch['index'][0]}.png" 
            if not batch['index'][0]==0 else 
            f"train-video/{self.true_global_step}.png",
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
                    "img": F.interpolate(
                        noise_out['noise'][:, :3],
                        (512,512),
                        mode="nearest",
                    ).permute(0,2,3,1)[0],
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                }
            ]
            + [
                {
                    "type": "grayscale",
                    "img": F.interpolate(
                        noise_out['det_mask'].float(),
                        (512,512),
                        mode="nearest",
                    ).permute(0,2,3,1)[0,:,:,0],
                    "kwargs": {"cmap": None, "data_range": (0., 1.)},
                },
            ]
            # + (
            #     [
            #         {
            #             "type": "grayscale",
            #             "img": F.interpolate(
            #                 noise_out['area_count'].float(),
            #                 (512,512),
            #                 mode="nearest",
            #             ).permute(0,2,3,1)[0,:,:,0],
            #             "kwargs": {"cmap": None, "data_range": (0, 16)},
            #         },
            #     ]
            #     if 'area_count' in noise_out
            #     else []
            # )
            # + (
            #     [
            #         {
            #             "type": "grayscale",
            #             "img": F.interpolate(
            #                 noise_out['z_mean_BCHW'].float(),
            #                 (512,512),
            #                 mode="nearest",
            #             ).permute(0,2,3,1)[0,:,:,0],
            #             "kwargs": {"data_range": (0.5, 1.5)},
            #         },
            #     ]
            #     if 'z_mean_BCHW' in noise_out
            #     else []
            # )
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
            name=f"validation_step_{batch['index'][0]}",
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
        # if isinstance(self.geometry, BaseImplicitGeometryGenerator):
        #     self.geometry.regenerate()
        self.t_scheduler.set_min_max_steps(0.4,0.4)
        out = self(batch, vis=True)
        noise_out = self.noise_generator(out, batch)
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]
        schedule_out = self.t_scheduler(guidance_inp.shape[0])
        if self.cfg.stage != "texture":
            try:
                guidance_out = self.guidance(
                    guidance_inp, 
                    self.prompt_utils, 
                    **batch, 
                    **schedule_out,
                    **noise_out,
                    rgb_as_latents=False, 
                    return_rgb_1step_orig=True,
                )
            except AttributeError:
                guidance_out = {}
        else:
            guidance_out = {}
        # metric evaluation
        if self.cfg.enable_eval_metirc:
            with torch.no_grad():
                prompt = self.prompt_utils.prompt
                clip_img = (out["comp_rgb"].permute(0,3,1,2)[0]*255).int()
                for mtc_name, metirc in self.metrics.items():
                    self.log(f"metric-test/{mtc_name}", metirc(clip_img, prompt), reduce_fx="max")
                    
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
                        "img": guidance_out["rgb_1step_orig"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "rgb_1step_orig" in guidance_out and self.cfg.stage != "texture"
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": F.interpolate(
                            noise_out['noise'][:, :3],
                            (512,512),
                            mode="nearest",
                        ).permute(0,2,3,1)[0],
                        "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out and self.cfg.stage != "texture"
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
                if "depth" in out and self.cfg.stage != "texture"
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        # threestudio.info(self.noise_generator.profiler.summary())
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