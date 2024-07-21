from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, get_device
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *
from threestudio.utils.rasterize import NVDiffRasterizerContext
import nvdiffrast.torch as dr


@threestudio.register("timestep-scheduler")
class TimestepScheduler(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # set >0 to turn on
        min_step_percent: Any = 0.02
        max_step_percent: Any = 0.98
        trainer_max_steps: int = 25000
        # HiFA configurations: https://hifa-team.github.io/HiFA-site/
        sqrt_anneal: bool = (
            False  # requires setting min_step_percent=0.3 to work properly
        )
        # Dreamtime configurations: https://arxiv.org/pdf/2306.12422
        dreamtime_anneal: Any = field(
            default_factory=lambda: [800,400,100,150]
        )
        linear_anneal_max: Any = None
        linear_anneal_min: Any = None
        linear_anneal_steps: Any = None
        
    cfg: Config
    
    def configure(self) -> None:
        super().configure()
        self.min_step: Optional[float] = None
        self.max_step: Optional[float] = None
        if self.cfg.dreamtime_anneal is not None:
            assert len(self.cfg.dreamtime_anneal)==4
            dreamtime_weight = np.ones(1000)
            dreamtime_idx = np.arange(0,1000,1)
            dreamtime_mask_1 = dreamtime_idx > self.cfg.dreamtime_anneal[0]
            dreamtime_mask_2 = dreamtime_idx < self.cfg.dreamtime_anneal[1]
            dreamtime_weight[dreamtime_mask_1] = np.exp(-(dreamtime_idx-self.cfg.dreamtime_anneal[0])**2/self.cfg.dreamtime_anneal[2]**2)[dreamtime_mask_1]
            dreamtime_weight[dreamtime_mask_2] = np.exp(-(dreamtime_idx-self.cfg.dreamtime_anneal[1])**2/self.cfg.dreamtime_anneal[3]**2)[dreamtime_mask_2]
            self.dt_w_normalized = 1.-np.cumsum(dreamtime_weight, axis=0)/np.sum(dreamtime_weight)
        if self.cfg.linear_anneal_max is not None and self.cfg.linear_anneal_min is not None and self.cfg.linear_anneal_steps is not None:
            assert len(self.cfg.linear_anneal_max) == len(self.cfg.linear_anneal_steps) and len(self.cfg.linear_anneal_min) == len(self.cfg.linear_anneal_steps) 
            self.linear_anneal_max = np.array(sorted(self.cfg.linear_anneal_max))
            self.linear_anneal_min = np.array(sorted(self.cfg.linear_anneal_min))
            self.linear_anneal_steps = np.array(sorted(self.cfg.linear_anneal_steps))
        
    def forward(self, batch_size):
        t_perc_ref = self.min_step+(self.max_step-self.min_step)*torch.rand(batch_size, device=self.device)
        schedule_out = {
            't_perc_ref': t_perc_ref
        }
        return schedule_out

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = min_step_percent
        self.max_step = max_step_percent
        
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413

        if self.cfg.dreamtime_anneal is not None:
            percentage = 1. - np.absolute(self.dt_w_normalized-float(global_step) / self.cfg.trainer_max_steps).argmin()/1000.
        elif self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
        elif self.cfg.linear_anneal_max is not None and self.cfg.linear_anneal_min is not None and self.cfg.linear_anneal_steps is not None:
            if np.all(global_step<self.linear_anneal_steps):
                x_i = 1
            elif not np.any(global_step<self.linear_anneal_steps):
                x_i = len(self.linear_anneal_steps)-1
            else:
                x_i = np.argmax(global_step<self.linear_anneal_steps)
            t_s, t_e = self.linear_anneal_steps[x_i-1], self.linear_anneal_steps[x_i]
            p_max_s, p_max_e = self.linear_anneal_max[x_i-1], self.linear_anneal_max[x_i]
            p_min_s, p_min_e = self.linear_anneal_min[x_i-1], self.linear_anneal_min[x_i]
            p_max = p_max_s * (t_e-global_step) / (t_e-t_s) + p_max_e * (global_step-t_s) / (t_e-t_s)
            p_min = p_min_s * (t_e-global_step) / (t_e-t_s) + p_min_e * (global_step-t_s) / (t_e-t_s)
            percentage = np.random.random() * (p_max-p_min) + p_min
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
            return
        if type(self.cfg.max_step_percent) not in [float, int]:
            max_step_percent = self.cfg.max_step_percent[1]
        else:
            max_step_percent = self.cfg.max_step_percent
        curr_percent = (
            max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
        ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
        self.set_min_max_steps(
            min_step_percent=curr_percent,
            max_step_percent=curr_percent,
        )

@threestudio.register("dev-timestep-scheduler")
class DevTimestepScheduler(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # set >0 to turn on
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        n_steps: int = 50
        start_percent: float = 0.3
        trainer_max_steps: int = 25000
        # HiFA configurations: https://hifa-team.github.io/HiFA-site/
        sqrt_anneal: bool = (
            True  # requires setting min_step_percent=0.3 to work properly
        )
        # Dreamtime configurations: https://arxiv.org/pdf/2306.12422
        dreamtime_anneal: Any = field(
            default_factory=lambda: [800,400,100,150]
        )
        
    cfg: Config
    
    def configure(self) -> None:
        super().configure()
        self.ref_tag = None
        self.t_perc_ref = self.cfg.max_step_percent
        self.t_perc_tgt = self.cfg.max_step_percent
        if self.cfg.dreamtime_anneal is not None:
            assert len(self.cfg.dreamtime_anneal)==4
            dreamtime_weight = np.ones(1000)
            dreamtime_idx = np.arange(0,1000,1)
            dreamtime_mask_1 = dreamtime_idx > self.cfg.dreamtime_anneal[0]
            dreamtime_mask_2 = dreamtime_idx < self.cfg.dreamtime_anneal[1]
            dreamtime_weight[dreamtime_mask_1] = np.exp(-(dreamtime_idx-self.cfg.dreamtime_anneal[0])**2/self.cfg.dreamtime_anneal[2]**2)[dreamtime_mask_1]
            dreamtime_weight[dreamtime_mask_2] = np.exp(-(dreamtime_idx-self.cfg.dreamtime_anneal[1])**2/self.cfg.dreamtime_anneal[3]**2)[dreamtime_mask_2]
            self.dt_w_normalized = 1.-np.cumsum(dreamtime_weight, axis=0)/np.sum(dreamtime_weight)
        
    def forward(self, batch_size):
        t_perc_ref = torch.full((batch_size,), self.t_perc_ref)
        t_perc_tgt = torch.full((batch_size,), self.t_perc_tgt)
        schedule_out = {
            't_perc_ref': t_perc_ref,
            't_perc_tgt': t_perc_tgt,
            'ref_tag': self.ref_tag,
        }
        return schedule_out
        
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        
        if self.cfg.dreamtime_anneal is not None:
            percentage = 1. - np.absolute(self.dt_w_normalized-float(global_step) / self.cfg.trainer_max_steps).argmin()/1000.
        elif self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5
        else:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            )
        ref_stage = int(percentage * (self.cfg.n_steps-1))
        percentage_ref = ref_stage/self.cfg.n_steps
        percentage_tgt = (ref_stage+1)/self.cfg.n_steps
        self.t_perc_ref = self.cfg.max_step_percent - percentage_ref * (self.cfg.max_step_percent-self.cfg.min_step_percent)
        self.t_perc_tgt = self.cfg.max_step_percent - percentage_tgt * (self.cfg.max_step_percent-self.cfg.min_step_percent)
        
        if self.t_perc_ref < self.cfg.start_percent:
            self.ref_tag = ref_stage
        else:
            self.ref_tag = None