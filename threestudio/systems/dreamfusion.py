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

@threestudio.register("dreamfusion-system")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        rgb_as_latents: bool = False
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
        render_out = self.renderer(**batch)
        if not vis and random.random() < self.normal_as_rgb_prob:
            if 'comp_normal' in render_out:
                render_out['comp_rgb'] = render_out['comp_normal'] 
            else:
                threestudio.warn('do not found normal in render output')
        if vis and self.cfg.rgb_as_latents:
            render_out["comp_rgb"] = self.guidance.vae_decode(
                self.guidance.pipe.vae,
                render_out["comp_rgb"].permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        if self.C(self.cfg.loss.lambda_normal_smooth) <= 0:
            self.material.requires_normal = False
            
        # set identity
        if isinstance(self.geometry, BaseImplicitGeometryGenerator):
            self.geometry.regenerate()
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

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z-variance loss proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                print(self.geometry.cfg.normal_type, self.material.requires_normal)
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            loss += (
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
            ) * self.C(self.cfg.loss.lambda_normal_smooth)
        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            loss += (normals - normals_perturb).abs().mean() * self.C(self.cfg.loss.lambda_3d_normal_smooth)

        
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
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        guidance_out = self.guidance(
            out["comp_rgb"], 
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
        schedule_out = self.t_scheduler(out["comp_rgb"].shape[0])
        try:
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
        self.save_image_grid(
            f"it{self.true_global_step}-test-rgb/{batch['index'][0]}.png",
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
            ),
            name="test_step_rgb",
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
        self.save_img_sequence(
            f"it{self.true_global_step}-test-rgb",
            f"it{self.true_global_step}-test-rgb",
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