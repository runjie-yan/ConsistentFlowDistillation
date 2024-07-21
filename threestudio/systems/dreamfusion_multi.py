from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.dreamfusion import DreamFusion
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("dreamfusion-multi-system")
class DreamFusionMulti(DreamFusion):
    @dataclass
    class Config(DreamFusion.Config):
        pass

    cfg: Config
    
    def configure(self):
        return super().configure()


    def validation_step(self, batch, batch_idx):
        for ptc_id in range(self.geometry.cfg.particle_num):
            self.geometry.regenerate(ptc_id)
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
            
            if not self.cfg.rgb_as_latents:
                enc_dec_ret = False
                try:
                    with torch.no_grad():
                        img_enc_dec = self.guidance.vae_decode(
                            self.guidance.pipe.vae, guidance_out['latents']
                        ).permute(0,2,3,1)
                        enc_dec_ret = True
                except:
                    pass
            self.save_image_grid(
                f"train-{ptc_id}/it{self.true_global_step}-{batch['index'][0]}.png" 
                if not batch['index'][0]==0 else 
                f"train-{ptc_id}-video/{self.true_global_step}.png",
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
                    if enc_dec_ret
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
                            "type": "grayscale",
                            "img": F.interpolate(
                                noise_out['area_count'].float(),
                                (512,512),
                                mode="nearest",
                            ).permute(0,2,3,1)[0,:,:,0],
                            "kwargs": {"cmap": None, "data_range": (0, 16)},
                        },
                    ]
                    if 'area_count' in noise_out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": F.interpolate(
                                noise_out['z_mean_BCHW'].float(),
                                (512,512),
                                mode="nearest",
                            ).permute(0,2,3,1)[0,:,:,0],
                            "kwargs": {"data_range": (0.5, 1.5)},
                        },
                    ]
                    if 'z_mean_BCHW' in noise_out
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
                name=f"validation_step_{batch['index'][0]}",
                step=self.true_global_step,
                # texts=(["rgb"] if "comp_rgb" in out else [])
                # + (["x_decode"] if enc_dec_ret else [])
                # + (["x_gt"] if "rgb_1step_orig" in guidance_out else [])
                # + ["noise"] + ["noise_det_mask"]
                # + (['area_count'] if 'area_count' in noise_out else [])
                # + (['z_mean_smooth'] if 'z_mean_BCHW' in noise_out else [])
                # + (["normal"] if "comp_normal" in out else [])
                # + (["z_mean"] if "z_mean" in out else [])
            )

    def test_step(self, batch, batch_idx):
        for ptc_id in range(self.geometry.cfg.particle_num):
            self.geometry.regenerate(ptc_id)
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
                               
            if not self.cfg.rgb_as_latents:
                enc_dec_ret = False
                try:
                    with torch.no_grad():
                        img_enc_dec = self.guidance.vae_decode(
                            self.guidance.pipe.vae, guidance_out['latents']
                        ).permute(0,2,3,1)
                        enc_dec_ret = True
                except:
                    pass
            self.save_image_grid(
                f"it{self.true_global_step}-{ptc_id}-test/{batch['index'][0]}.png",
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
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if enc_dec_ret
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
                            "type": "grayscale",
                            "img": F.interpolate(
                                noise_out['area_count'].float(),
                                (512,512),
                                mode="nearest",
                            ).permute(0,2,3,1)[0,:,:,0],
                            "kwargs": {"cmap": None, "data_range": (0, 16)},
                        },
                    ]
                    if 'area_count' in noise_out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": F.interpolate(
                                noise_out['z_mean_BCHW'].float(),
                                (512,512),
                                mode="nearest",
                            ).permute(0,2,3,1)[0,:,:,0],
                            "kwargs": {"data_range": (0.5, 1.5)},
                        },
                    ]
                    if 'z_mean_BCHW' in noise_out
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
                # texts=(["rgb"] if "comp_rgb" in out else [])
                # + (["x_decode"] if enc_dec_ret else [])
                # + (["x_gt"] if "rgb_1step_orig" in guidance_out else [])
                # + ["noise"] + ["noise_det_mask"]
                # + (['area_count'] if 'area_count' in noise_out else [])
                # + (['z_mean_smooth'] if 'z_mean_BCHW' in noise_out else [])
                # + (["normal"] if "comp_normal" in out else [])
                # + (["z_mean"] if "z_mean" in out else [])
            )

    def on_test_epoch_end(self):
        for ptc_id in range(self.geometry.cfg.particle_num):
            self.save_img_sequence(
                f"it{self.true_global_step}-{ptc_id}-test",
                f"it{self.true_global_step}-{ptc_id}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step,
            )
            try:
                self.save_img_sequence(
                    f"it{self.true_global_step}-{ptc_id}-train",
                    f"train-{ptc_id}-video",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=10,
                    name="train",
                    step=self.true_global_step,
                )
            except:
                threestudio.warn("can not generate training video")