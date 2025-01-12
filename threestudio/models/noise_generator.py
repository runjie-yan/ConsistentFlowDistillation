from dataclasses import dataclass

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, get_device
from threestudio.utils.ops import scale_tensor
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("noise-generator")
class NoiseGenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        """
        sqrt_beta: enable determined noise
        set >0 to turn on
        set <1 to balance between random and determined noise
        set =1 to use fully determined noise
        """

        sqrt_beta: float = 0.0
        noise_h: int = 64
        noise_w: int = 64
        noise_c: int = 4
        # weight_decay = gamma, corresponds to SDE guidance
        weight_decay: Any = 0.0

    cfg: Config

    def configure(self) -> None:
        self.noise_background = nn.Parameter(
            torch.randn(
                1,
                self.cfg.noise_c,
                self.cfg.noise_h,
                self.cfg.noise_w,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.sqrt_beta = 0.0
        super().configure()

    @torch.no_grad()
    def get_determined_noise(
        self, out, batch
    ) -> Tuple[Float[Tensor, "1 C H W"], Float[Tensor, "1 C H W"]]:
        # only need to consider for B=1, first is determined part, second is its mask
        return (
            self.noise_background.clone(),
            torch.ones_like(self.noise_background).bool(),
            {},
        )

    @torch.no_grad()
    def get_noise_(self, out, batch, idx):
        # determined noise methods support for 1 batch size only
        # TODO: use chunck batch
        batch_ = {}
        for k, v in batch.items():
            try:
                batch_[k] = v[idx : idx + 1]
            except:
                batch_[k] = v
        out_ = {}
        for k, v in out.items():
            try:
                out_[k] = v[idx : idx + 1]
            except:
                out_[k] = v

        base_noise = torch.randn(
            1, self.cfg.noise_c, self.cfg.noise_h, self.cfg.noise_w, device=self.device
        )
        if self.sqrt_beta <= 0.0:
            # random noise
            return base_noise, torch.zeros_like(base_noise).bool(), {}
        else:
            det_noise, mask, eval_utils = self.get_determined_noise(out_, batch_)
            base_noise[mask] = (
                det_noise[mask] * self.sqrt_beta
                + base_noise[mask] * (1 - self.sqrt_beta**2) ** 0.5
            )
            return base_noise, mask, eval_utils

    @torch.no_grad()
    def __call__(self, out, batch):
        if "rays_o" in batch:
            b = batch["rays_o"].shape[0]
        elif "batch_size" in batch:
            b = batch["batch_size"]
        noises = []
        masks = []
        eval_utils = []
        for i in range(b):
            noise, mask, eval_util = self.get_noise_(out, batch, i)
            noises.append(noise)
            masks.append(mask)
            eval_utils.append(eval_util)
        noises = torch.cat(noises, dim=0)
        masks = torch.cat(masks, dim=0)
        out = {"noise": noises, "det_mask": masks}
        for k in eval_utils[0].keys():
            items = []
            for eu in eval_utils:
                items.append(eu[k])
            items = torch.cat(items, dim=0)
            out.update({k: items})
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.sqrt_beta = C(self.cfg.sqrt_beta, epoch, global_step)
        self.weight_decay = C(self.cfg.weight_decay, epoch, global_step)
        if self.training:
            self.noise_background.data = (
                self.noise_background.data * (1-self.weight_decay) ** 0.5
                + torch.randn_like(self.noise_background.data)
                * self.weight_decay ** 0.5
            )


@threestudio.register("worldmap-noise-generator")
class WorldmapNoiseGenerator(NoiseGenerator):
    @dataclass
    class Config(NoiseGenerator.Config):
        noise_background_fix: bool = True
        noise_theta: float = 180.0

    cfg: Config

    def configure(self):
        self.noise_buffer_width = int((360 / self.cfg.noise_theta) * self.cfg.noise_w)
        self.noise_buffer_height = int((180 / self.cfg.noise_theta) * self.cfg.noise_h)
        assert (
            self.cfg.noise_w <= self.noise_buffer_width
        ), "the buffer size is too small"
        # prepare noise buffer with periodic padding
        noise_buffer = torch.randn(
            1,
            self.cfg.noise_c,
            self.noise_buffer_height + self.cfg.noise_h,
            self.noise_buffer_width + self.cfg.noise_w,
            device=self.device,
        )
        noise_buffer[..., : self.cfg.noise_w // 2] = noise_buffer[
            ...,
            self.noise_buffer_width : self.noise_buffer_width + self.cfg.noise_w // 2,
        ]
        noise_buffer[..., -self.cfg.noise_w // 2 :] = noise_buffer[
            ..., self.cfg.noise_w // 2 : self.cfg.noise_w
        ]
        self.noise_buffer = nn.Parameter(noise_buffer, requires_grad=False)
        super().configure()

    @torch.no_grad()
    def get_determined_noise(self, out, batch):
        b, h, w, _ = out["comp_rgb"].shape
        elevation, azimuth, opacity = (
            batch["elevation"],
            batch["azimuth"],
            out["opacity"],
        )
        assert b == 1, "only support for one batch"

        elevation = (
            (-elevation[0] * self.noise_buffer_height / 180) % self.noise_buffer_height
        ).int()
        azimuth = (
            (azimuth[0] * self.noise_buffer_width / 360) % self.noise_buffer_width
        ).int()
        noise = self.noise_buffer[
            ...,
            elevation : elevation + self.cfg.noise_h,
            azimuth : azimuth + self.cfg.noise_w,
        ].clone()
        det_mask = torch.ones_like(noise).bool()
        opacity_mean_pool = nn.AvgPool2d(
            *[
                [
                    h // self.cfg.noise_h,
                    w // self.cfg.noise_w,
                ]
            ]
            * 2
        )
        mask = (
            opacity_mean_pool(
                opacity.repeat(1, 1, 1, self.cfg.noise_c).permute(0, 3, 1, 2)
            )
            < 0.5
        )
        if self.cfg.noise_background_fix:
            noise[mask] = self.noise_background[mask]
        else:
            det_mask[mask] = False
        return noise, det_mask, {}
    
    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        super().update_step(epoch, global_step, on_load_weights)
        if self.training:
            # SDE noise injection
            self.noise_buffer.data = (
                self.noise_buffer.data
                * (1.0 - self.weight_decay) ** 0.5
                + torch.randn_like(self.noise_buffer.data)
                * self.weight_decay ** 0.5
            )


@threestudio.register("dev-noise-generator")
class DevNoiseGenerator(NoiseGenerator):
    @dataclass
    class Config(NoiseGenerator.Config):
        mode: int = 0

    cfg: Config

    def configure(self):
        noise_buffer = torch.randn(
            1, self.cfg.noise_c, self.cfg.noise_h, self.cfg.noise_w, device=self.device
        )
        self.noise_buffer = nn.Parameter(noise_buffer, requires_grad=False)
        super().configure()

    @torch.no_grad()
    def get_determined_noise(self, out, batch):
        noise = torch.randn(
            1,
            self.cfg.noise_c,
            self.cfg.noise_h,
            self.cfg.noise_w,
            device=self.device,
        )
        if self.cfg.mode == 0:
            noise[
                :,
                :,
                self.cfg.noise_h // 4 : 3 * self.cfg.noise_h // 4,
                self.cfg.noise_w // 4 + 8 : 3 * self.cfg.noise_w // 4 + 8,
            ] = self.noise_buffer[
                :,
                :,
                self.cfg.noise_h // 4 : 3 * self.cfg.noise_h // 4,
                self.cfg.noise_w // 4 + 8 : 3 * self.cfg.noise_w // 4 + 8,
            ]

            mask = torch.zeros_like(noise).bool()
            mask[
                :,
                :,
                self.cfg.noise_h // 4 : 3 * self.cfg.noise_h // 4,
                self.cfg.noise_w // 4 + 8 : 3 * self.cfg.noise_w // 4 + 8,
            ] = ~mask[
                :,
                :,
                self.cfg.noise_h // 4 : 3 * self.cfg.noise_h // 4,
                self.cfg.noise_w // 4 + 8 : 3 * self.cfg.noise_w // 4 + 8,
            ]
        elif self.cfg.mode == 1:
            noise[
                :,
                :,
                :,
                self.cfg.noise_w // 2 :,
            ] = self.noise_buffer[
                :,
                :,
                :,
                self.cfg.noise_w // 2 :,
            ]

            mask = torch.zeros_like(noise).bool()
            mask[
                :,
                :,
                :,
                self.cfg.noise_w // 2 :,
            ] = True
        return noise, mask, {}


# The multi-view consistent noise generator
@threestudio.register("triplane-noise-generator")
class TriplaneNoiseGenerator(NoiseGenerator):
    @dataclass
    class Config(NoiseGenerator.Config):
        radius: float = 1.0
        noise_background_fix: bool = True
        separate: bool = True
        apply_transformation: Optional[str] = None

        opc_thres: float = 0.1
        noise_ptc_resolution: int = 2048
        context_type: str = "gl"
        peel_depth: int = 4
        debug: bool = False

        sph_warmup_steps: int = -1
        sph_warmup_radius: float = 0.5

        smooth_regularization: bool = False

        half: bool = False # not used

    cfg: Config

    def configure(self):
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)
        # bounding box
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
        # noise particles
        self.noise_background = nn.Parameter(
            torch.randn(
                1,
                self.cfg.noise_c,
                self.cfg.noise_h,
                self.cfg.noise_w,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.noise_ptc_val = nn.Parameter(
            torch.randn(
                6,
                self.cfg.peel_depth,
                self.cfg.noise_ptc_resolution,
                self.cfg.noise_ptc_resolution,
                self.cfg.noise_c,
                device=self.device,
            ),
            requires_grad=False,
        )  # 6 Hp Wp C
        self.kernel_size = 1

        # sph warm up (regularize shape)
        # not used
        self.sph_warmup = False
        super().configure()

    @torch.no_grad()
    def get_determined_noise(self, out, batch):
        b, h, w, _ = out["comp_rgb"].shape
        s_h, s_w = h // self.cfg.noise_h, w // self.cfg.noise_w  # stride
        k_h, k_w = (s_h + 1) // 2 * 2, (s_w + 1) // 2 * 2  # kernel size
        assert b == 1, "only support for one batch"
        assert (
            h % self.cfg.noise_h == 0 and w % self.cfg.noise_w == 0
        ), "no support for strange shape"
        opacity: Float[Tensor, "Hi Wi 1"] = out["opacity"][0]
        z_mean: Float[Tensor, "Hi Wi 1"] = out["z_mean"][0]
        rays_d: Float[Tensor, "Hi Wi 3"] = batch["rays_d"][0]
        rays_o: Float[Tensor, "Hi Wi 3"] = batch["rays_o"][0]
        suf_pts: Float[Tensor, "Hi Wi 3"] = z_mean * rays_d + rays_o
        # scale to clip space
        suf_pts = scale_tensor(suf_pts, self.bbox, (-1, 1))

        # prepare finial noise
        if self.cfg.noise_background_fix:
            noise_: Float[Tensor, "H W C"] = self.noise_background.clone().permute(
                0, 2, 3, 1
            )[0]
            det_mask = torch.ones_like(noise_).bool()
        else:
            noise_: Float[Tensor, "H W C"] = torch.randn(
                self.cfg.noise_h,
                self.cfg.noise_w,
                self.cfg.noise_c,
                device=self.device,
            )
            det_mask = torch.zeros_like(noise_).bool()

        # compute corner vertex values
        if self.sph_warmup:
            # assume geometry is a shpere
            mean_pool_pix = nn.AvgPool2d(
                (k_h, k_w), (s_h, s_w), padding=(k_h // 2, k_w // 2)
            )
            rays_o_pix_BCHW: Float[Tensor, "1 3 Hi Wi"] = rays_o[None].permute(
                0, 3, 1, 2
            )
            rays_d_pix_BCHW: Float[Tensor, "1 3 Hi Wi"] = rays_d[None].permute(
                0, 3, 1, 2
            )

            rays_o_mean_BCHW: Float[Tensor, "1 3 H+1 W+1"] = mean_pool_pix(
                rays_o_pix_BCHW
            )
            rays_d_mean_BCHW: Float[Tensor, "1 3 H+1 W+1"] = mean_pool_pix(
                rays_d_pix_BCHW
            )

            neg_b: Float[Tensor, "1 1 H+1 W+1"] = -(
                rays_o_mean_BCHW * rays_d_mean_BCHW
            ).sum(dim=1, keepdim=True)
            neg_b_mask = neg_b > 0.0
            b2_minus_4a_dot_c: Float[Tensor, "1 1 H+1 W+1"] = (
                neg_b**2
                - (rays_o_mean_BCHW**2).sum(dim=1, keepdim=True)
                + self.cfg.sph_warmup_radius**2
            )
            b2_minus_4a_dot_c_mask = b2_minus_4a_dot_c > 0.0

            corner_suf_pts: Float[Tensor, "(H+1)(W+1) 3"] = (
                torch.nan_to_num(
                    rays_o_mean_BCHW
                    + rays_d_mean_BCHW * (neg_b - b2_minus_4a_dot_c.sqrt())
                )
                .permute(0, 2, 3, 1)
                .reshape(-1, 3)
            )
            corner_opacity: Float[Tensor, "(H+1)(W+1) 1"] = (
                torch.logical_and(neg_b_mask, b2_minus_4a_dot_c_mask)
                .float()
                .permute(0, 2, 3, 1)
                .reshape(-1, 1)
            )
        elif self.cfg.smooth_regularization:
            # using kernel smoothing...
            mean_pool_pix = nn.AvgPool2d(
                (k_h, k_w), (s_h, s_w), padding=(k_h // 2, k_w // 2)
            )
            opacity_pix_BCHW: Float[Tensor, "1 1 Hi Wi"] = opacity[None].permute(
                0, 3, 1, 2
            )
            point_pix_BCHW: Float[Tensor, "1 3 Hi Wi"] = suf_pts[None].permute(
                0, 3, 1, 2
            )
            mask_pix_BCHW: Bool[Tensor, "1 1 Hi Wi"] = (
                opacity_pix_BCHW > self.cfg.opc_thres
            )

            # ignore background points when averaging
            point_pix_BCHW[~mask_pix_BCHW.repeat(1, 3, 1, 1)] = 0.0

            point_mean_BCHW = mean_pool_pix(point_pix_BCHW)
            area_BCHW = mean_pool_pix(mask_pix_BCHW.float())
            opacity_BCHW: Float[Tensor, "1 1 H+1 W+1"] = mean_pool_pix(
                opacity_pix_BCHW
            )
            point_mean_BCHW = torch.nan_to_num(point_mean_BCHW / area_BCHW)

            corner_opacity: Float[Tensor, "(H+1)(W+1) 1"] = opacity_BCHW.permute(
                0, 2, 3, 1
            ).reshape(-1, 1)
            corner_suf_pts: Float[Tensor, "(H+1)(W+1) 3"] = point_mean_BCHW.permute(
                0, 2, 3, 1
            ).reshape(-1, 3)
        else:
            # get 4 2d indices of the corner of the noise pixel on the image map
            corner_line_idx_h = torch.arange(0, h + 1, s_h, device=self.device)
            corner_line_idx_w = torch.arange(0, w + 1, s_w, device=self.device)
            corner_line_idx_h[-1] = h - 1  # last idx is out of the picture
            corner_line_idx_w[-1] = w - 1  # last idx is out of the picture

            # select the corner points of noise pixels
            corner_idx: Int[Tensor, "H+1 W+1 2"] = torch.stack(
                torch.meshgrid(corner_line_idx_h, corner_line_idx_w, indexing="ij"),
                dim=-1,
            )
            corner_mask: Bool[Tensor, "(H+1)(W+1) 3"] = torch.zeros(
                h, w, device=self.device
            ).bool()
            corner_mask[corner_idx[..., 0], corner_idx[..., 1]] = 1

            corner_opacity: Float[Tensor, "(H+1)(W+1) 1"] = opacity[corner_mask]
            corner_suf_pts: Float[Tensor, "(H+1)(W+1) 3"] = suf_pts[corner_mask]

        # creat faces for rasterization
        indices = torch.arange(
            self.cfg.noise_h * (self.cfg.noise_w + 1), device=self.device
        )
        mask = (indices + 1) % (self.cfg.noise_w + 1) != 0
        faces_line_idx = indices[mask]
        faces: Int[Tensor, "2HW 3"] = torch.cat(
            [
                torch.stack(
                    [
                        faces_line_idx,
                        faces_line_idx + 1,
                        faces_line_idx + (self.cfg.noise_w + 1),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        faces_line_idx + 1,
                        faces_line_idx + (self.cfg.noise_w + 1),
                        faces_line_idx + (self.cfg.noise_w + 1) + 1,
                    ],
                    dim=-1,
                ),
            ],
            dim=0,
        ).long()

        # decide back/foreground mask
        concrete_mask: Bool[Tensor, "(H+1)(W+1)"] = (
            corner_opacity > self.cfg.opc_thres
        ).squeeze(dim=-1)
        foreground_mask: Bool[Tensor, "2HW"] = concrete_mask[faces].all(dim=-1)

        # rasterization
        # do reduction
        for mask, pts in [
            (foreground_mask, corner_suf_pts),
        ]:
            # position 0 is the homeless
            area_count_sub: Float[Tensor, "Nm"] = torch.zeros(
                faces[mask].shape[0], device=self.device
            )
            noise_sumed_sub: Float[Tensor, "Nm C"] = torch.zeros(
                faces[mask].shape[0], self.cfg.noise_c, device=self.device
            )
            for pmu, sgn, noise_idx in zip(
                [
                    [
                        0,
                        1,
                        2,
                    ],
                    [
                        0,
                        1,
                        2,
                    ],
                    [
                        2,
                        0,
                        1,
                    ],
                    [
                        2,
                        0,
                        1,
                    ],
                    [
                        1,
                        2,
                        0,
                    ],
                    [
                        1,
                        2,
                        0,
                    ],
                ],
                [+1, -1] * 3,
                range(6),
            ):
                # if self.cfg.half and noise_idx % 2 == 1:
                #     continue
                # change to homocoordinate
                pts_xyz = pts[:, pmu]
                pts_xyz: Float[Tensor, "(H+1)(W+1) 3"]

                # IMPORTANT: apply some manifold transformation
                if self.cfg.apply_transformation is not None:
                    if self.cfg.apply_transformation == "sph2sqr":
                        # pts_xyz[:, 2] = sgn * pts_xyz[:, 2]
                        pts_theta = torch.atan2(
                            torch.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2),
                            pts_xyz[:, 2].abs(),
                        )
                        pts_phi = torch.atan2(pts_xyz[:, 0], pts_xyz[:, 1])
                        pts_r = sgn * torch.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2 + pts_xyz[:, 2] ** 2)
                        sph2sqr_mask_0 = torch.logical_and(
                            pts_phi > 0.0, pts_phi <= torch.pi / 2
                        )
                        sph2sqr_mask_1 = torch.logical_and(
                            pts_phi > torch.pi / 2, pts_phi <= torch.pi
                        )
                        sph2sqr_mask_2 = torch.logical_and(
                            pts_phi > -torch.pi, pts_phi <= -torch.pi / 2
                        )
                        sph2sqr_mask_3 = torch.logical_and(
                            pts_phi > -torch.pi / 2, pts_phi <= 0.0
                        )
                        sph2sqr_d = torch.sqrt(1.0 - torch.cos(pts_theta))
                        pts_xyz[sph2sqr_mask_0, 0] = sph2sqr_d[sph2sqr_mask_0]
                        pts_xyz[sph2sqr_mask_0, 1] = sph2sqr_d[sph2sqr_mask_0] * (
                            pts_phi[sph2sqr_mask_0] * 2.0 / (torch.pi / 2.0) - 1.0
                        )
                        pts_xyz[sph2sqr_mask_1, 0] = -sph2sqr_d[sph2sqr_mask_1] * (
                            (pts_phi[sph2sqr_mask_1] - torch.pi / 2)
                            * 2.0
                            / (torch.pi / 2.0)
                            - 1.0
                        )
                        pts_xyz[sph2sqr_mask_1, 1] = sph2sqr_d[sph2sqr_mask_1]
                        pts_xyz[sph2sqr_mask_2, 0] = -sph2sqr_d[sph2sqr_mask_2]
                        pts_xyz[sph2sqr_mask_2, 1] = -sph2sqr_d[sph2sqr_mask_2] * (
                            (pts_phi[sph2sqr_mask_2] + torch.pi)
                            * 2.0
                            / (torch.pi / 2.0)
                            - 1.0
                        )
                        pts_xyz[sph2sqr_mask_3, 0] = sph2sqr_d[sph2sqr_mask_3] * (
                            (pts_phi[sph2sqr_mask_3] + torch.pi / 2)
                            * 2.0
                            / (torch.pi / 2.0)
                            - 1.0
                        )
                        pts_xyz[sph2sqr_mask_3, 1] = -sph2sqr_d[sph2sqr_mask_3]
                        
                        pts_xyz[:,2] = pts_r
                    else:
                        raise ValueError
                # IMPORTANT: apply some manifold transformation

                pts_homo: Float[Tensor, "(H+1)(W+1) 4"] = torch.cat(
                    [pts_xyz, torch.ones_like(pts_xyz[:, :1])], dim=-1
                )
                PN = self.cfg.peel_depth
                for peel_i in range(self.cfg.peel_depth):
                    if peel_i == PN-1:
                        sp_mask: Bool[Tensor, "Nm"] = (pts_homo[:, 2][faces[mask]].abs() > (PN-1)/PN).all(dim=-1)
                    else:
                        sp_mask: Bool[Tensor, "Nm"] = torch.logical_and(
                            (pts_homo[:, 2][faces[mask]].abs() > peel_i/PN).all(dim=-1),
                            (pts_homo[:, 2][faces[mask]].abs() <= (peel_i+1)/PN).all(dim=-1),
                        )
                        
                    if self.cfg.separate:
                        if not sp_mask.any():
                            # skip empty layer
                            continue
                    else:
                        sp_mask: Bool[Tensor, "Nm"] = torch.ones_like(sp_mask).bool()

                    noise_map_val = self.noise_ptc_val[noise_idx][peel_i].flatten(
                        start_dim=0, end_dim=-2
                    )
                    # call cfg can waste a lot of time
                    noise_c_ = self.cfg.noise_c

                    # map points to triplane (multi-layers are considered)
                    with dr.DepthPeeler(
                        self.ctx.ctx,
                        pts_homo[None],
                        faces[mask][sp_mask].int(),
                        (self.cfg.noise_ptc_resolution, self.cfg.noise_ptc_resolution),
                    ) as peeler:
                        try:
                            rast, _ = peeler.rasterize_next_layer()
                        except RuntimeError:
                            threestudio.warn(
                                "Noise generator failed to rasterize_next_layer"
                            )
                            break
                        face_map: Int[Tensor, "Hp Wp"] = rast[0, :, :, 3].long()
                        if face_map.max() == 0:
                            # all faces are rasterized
                            break

                        # # DEBUG
                        # if pi == 0 and self.cfg.debug:
                        #     from time import time

                        #     import matplotlib.pyplot as plt
                        #     import numpy as np

                        #     debug_face_map = face_map.cpu().numpy()
                        #     debug_num = debug_face_map.max()
                        #     debug_colormap = plt.cm.get_cmap("tab20", debug_num)
                        #     debug_colored_image = debug_colormap(debug_face_map)
                        #     plt.imshow(debug_colored_image)
                        #     plt.axis("off")
                        #     plt.savefig(f"outputs/debug/{time()}.png", dpi=500)
                        #     plt.close()

                        flat_face_map = face_map.view(-1)
                        valid_face_mask = flat_face_map != 0
                        valid_face_map = (
                            flat_face_map[valid_face_mask] - 1
                        )  # remove 1 offset

                        index_1d = valid_face_map
                        index_2d = valid_face_map.unsqueeze(1).expand(-1, noise_c_)

                        # Compute src tensors
                        src_1d = torch.ones_like(valid_face_map, dtype=torch.float)
                        src_2d = noise_map_val[valid_face_mask]

                        # Compute scatter_reduce operations
                        area_count_sub[sp_mask] = area_count_sub[
                            sp_mask
                        ].scatter_reduce(
                            dim=0,
                            index=index_1d,
                            src=src_1d,
                            reduce="sum",
                        )

                        noise_sumed_sub[sp_mask] = noise_sumed_sub[
                            sp_mask
                        ].scatter_reduce(
                            dim=0,
                            index=index_2d,
                            src=src_2d,
                            reduce="sum",
                        )

            # map subset of faces back to noise space
            area_count = torch.zeros(
                2 * self.cfg.noise_h * self.cfg.noise_w, device=self.device
            )
            noise_sumed = torch.zeros(
                2 * self.cfg.noise_h * self.cfg.noise_w,
                self.cfg.noise_c,
                device=self.device,
            )
            area_count[mask] = area_count_sub
            noise_sumed[mask] = noise_sumed_sub
            area_count: Int[Tensor, "H W"] = area_count.reshape(
                2, self.cfg.noise_h, self.cfg.noise_w
            ).sum(dim=0)
            noise_sumed: Float[Tensor, "H W C"] = noise_sumed.reshape(
                2, self.cfg.noise_h, self.cfg.noise_w, self.cfg.noise_c
            ).sum(dim=0)

            # case: get no noise
            gt0_mask = area_count > 0  # use random noise
            det_mask[gt0_mask] = True

            # compute averaged noise by sum_i n_i / sqrt(|n|)
            noise_[gt0_mask] = (
                noise_sumed[gt0_mask] / area_count[gt0_mask].sqrt()[..., None]
            )

        noise = noise_[None, ...].permute(0, 3, 1, 2)
        det_mask = det_mask[None, ...].permute(0, 3, 1, 2)
        area_count = area_count[None, None, ...]
        eval_utils = {
            "area_count": area_count,
        }
        return noise, det_mask, eval_utils

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        super().update_step(epoch, global_step, on_load_weights)
        if self.cfg.sph_warmup_steps < global_step:
            self.sph_warmup = False
        else:
            self.sph_warmup = True
        if self.training:
            # SDE noise injection
            self.noise_background.data = (
                self.noise_background.data
                * (1.0 - self.weight_decay) ** 0.5
                + torch.randn_like(self.noise_background.data)
                * self.weight_decay ** 0.5
            )
            self.noise_ptc_val.data = (
                self.noise_ptc_val.data
                * (1.0 - self.weight_decay) ** 0.5
                + torch.randn_like(self.noise_ptc_val.data)
                * self.weight_decay ** 0.5
            )
