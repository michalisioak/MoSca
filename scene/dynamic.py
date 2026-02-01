# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
import colorsys
from typing import Any, Dict
import numpy as np
import torch
from torch import nn
import logging
from motion_scaffold.mosca import MoSca
from scene.cfg import GaussianSplattingConfig


class DynamicScene(nn.Module):
    def __init__(
        self,
        xyz: torch.Tensor,  # N,3
        rgbs: torch.Tensor,  # N,3
        scales: torch.Tensor,  # N,3
        scf: MoSca,
        cfg: GaussianSplattingConfig,
        quats: torch.Tensor | None = None,  # N,4
        opacities: torch.Tensor | None = None,  # N
    ):
        super().__init__()
        self.cfg = cfg
        self.scf = scf
        self.xyz = nn.Parameter(xyz)
        self.scales = nn.Parameter(scales)
        self.opacities = nn.Parameter(
            torch.ones(xyz.shape[0]) if opacities is None else opacities
        )
        self.quats = nn.Parameter(
            torch.ones((xyz.shape[0], 4)) if quats is None else quats
        )
        self.params = nn.ParameterDict(
            {
                "means": self.xyz,
                "scales": self.scales,
                "opacities": self.opacities,
                "quats": self.quats,
            }
        )
        optimizer_class = cfg.get_optimizer()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "means": optimizer_class(
                params=self.xyz,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "scales": optimizer_class(
                params=self.scales,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "opacities": optimizer_class(
                params=self.opacities,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
            "quats": optimizer_class(
                params=self.quats,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
        }
        self.strategy = cfg.strategy
        self.strategy.check_sanity(self.params, self.optimizers)
        self.state = self.strategy.initialize_state()

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        init_mean = ckpt["_xyz"]
        init_rgb = ckpt["_features_dc"]
        init_s = ckpt["_scaling"]
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

    def forward(self, t: int, active_sph_order=None):
        assert t < self.scf.T, "t is out of range!"
        sph_dim = 3 * sph_order2nfeat(active_sph_order)
        sph = self.get_c[:, :sph_dim]

        if torch.is_inference_mode_enabled():
            mu_live, fr_live = self.scf.fast_warp(
                target_tid=t,
                # all below are baked
                sk_ind=self.baked_sk_ind,
                sk_w=self.baked_sk_w,
                sk_ref_node_xyz=self.baked_sk_ref_node_xyz,
                sk_ref_node_quat=self.baked_sk_ref_node_quat,
                dyn_o=self.baked_dyn_o,
                query_xyz=self.baked_query_xyz,
                query_dir=self.baked_query_dir,
            )
        else:
            mu_live, fr_live = self.scf.warp(
                attach_node_ind=self.attach_ind,
                query_xyz=self.get_xyz(),
                query_dir=self.get_R_mtx(),
                query_tid=self.ref_time,
                target_tid=t,
                skinning_w_corr=(
                    self._skinning_weight if self.w_correction_flag else None
                ),
                dyn_o_flag=self.dyn_o_flag,
            )
        return mu_live, fr_live, s, o, sph

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
