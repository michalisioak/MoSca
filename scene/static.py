# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
import colorsys
from typing import Any, Dict
import numpy as np
import torch
from torch import nn
import logging
from scene.cfg import GaussianSplattingConfig


class StaticScene(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,  # N,3
        rgbs: torch.Tensor,  # N,3
        scales: torch.Tensor,  # N,3
        cfg: GaussianSplattingConfig,
        quats: torch.Tensor | None = None,  # N,4
        opacities: torch.Tensor | None = None,  # N
    ):
        super().__init__()
        self.cfg = cfg
        self.means = nn.Parameter(means)
        self.scales = nn.Parameter(scales)
        self.opacities = nn.Parameter(
                    torch.ones(means.shape[0]) if opacities is None else opacities
                )
        self.quats = nn.Parameter(
                    torch.ones((means.shape[0], 4)) if quats is None else quats
                )
        optimizer_class = cfg.get_optimizer()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "means": optimizer_class(
                params=self.params["means"],
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "scales": optimizer_class(
                params=self.params["scales"],
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "opacities": optimizer_class(
                params=self.params["opacities"],
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "quats": optimizer_class(
                params=self.params["quats"],
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
        }

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        init_mean = ckpt["xyz"]
        init_rgb = ckpt["features_dc"]
        init_s = ckpt["scaling"]
        max_sph_order = ckpt["max_sph_order"]
        model = cls(
            init_mean=init_mean,
            init_rgb=init_rgb,
            init_s=init_s,
            max_sph_order=max_sph_order,
        )
        model.load_state_dict(ckpt, strict=True)
        # ! important, must re-init the activation functions
        logging.info(
            f"Resume: Max scale: {model.max_scale}, Min scale: {model.min_scale}, Max sph order: {model.max_sph_order}"
        )
        model._init_act(model.max_scale, model.min_scale)
        return model

    def summary(self):
        logging.info(f"StaticGaussian: {len(self.xyz) / 1000.0:.1f}K points")
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel() / 1e6:.3f}M")
        logging.info("-" * 30)

    @torch.no_grad()
    def get_cate_color(self, color_plate=None, perm=None):
        gs_group_id = self.get_group
        unique_grouping = torch.unique(gs_group_id).sort()[0]
        if not hasattr(self, "group_colors"):
            if color_plate is None:
                n_cate = len(self.group_id.unique())
                hue = np.linspace(0, 1, n_cate + 1)[:-1]
                color_plate = torch.Tensor(
                    [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue]
                ).to(self.device)
            self.group_colors = color_plate
            self.group_sphs = RGB2SH(self.group_colors)
        if perm is None:
            perm = torch.arange(len(unique_grouping))

        cate_sph = torch.zeros(self.N, 3).to(self.device)
        index_color_map = {}
        for ind in perm:
            gid = unique_grouping[ind]
            cate_sph[gs_group_id == gid] = self.group_sphs[ind].unsqueeze(0)
            index_color_map[gid] = self.group_colors[ind]
        return cate_sph, index_color_map

    def forward(self, active_sph_order=None):
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        else:
            assert active_sph_order <= self.max_sph_order
        frame = self.get_R
        s = self.get_s
        o = self.get_o

        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c
        sph = sph[:, :sph_dim]

        if self.return_cate_colors_flag:
            # logging.warning(f"VIZ purpose, return the cate-color")
            cate_sph, _ = self.get_cate_color()
            sph = torch.zeros_like(sph)
            sph[..., :3] = cate_sph  # zero pad

        return self.xyz, frame, s, o, sph

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_pre_backward(
            self.params, self.optimizers, self.state, step, info
        )

    def step_post_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_post_backward(
            self.params, self.optimizers, self.state, step, info
        )  # pyright: ignore[reportCallIssue]

    def load(self, ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(torch.as_tensor(ckpt["xyz"], dtype=torch.float32))

        self._features_dc = nn.Parameter(
            torch.as_tensor(ckpt["features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(ckpt["features_rest"], dtype=torch.float32)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(ckpt["opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(ckpt["scaling"], dtype=torch.float32)
        )
        self._rotation = nn.Parameter(
            torch.as_tensor(ckpt["rotation"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(ckpt["max_radii2D"], dtype=torch.float32)
        # load others
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        return
