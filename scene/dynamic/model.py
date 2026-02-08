# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
from typing import Any, Dict
import torch
from torch import nn
import logging
from motion_scaffold import MoSca
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

    def summary(self):
        logging.info(f"StaticGaussian: {len(self.xyz) / 1000.0:.1f}K points")
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel() / 1e6:.3f}M")
        logging.info("-" * 30)

    def forward(self, t: int):
        assert t < self.scf.T, "t is out of range!"

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
        return mu_live, fr_live, self.scales, self.opacities, self.shs

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_pre_backward(
            self.params, self.optimizers, self.state, step, info
        )

    def step_post_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_post_backward(
            self.params, self.optimizers, self.state, step, info
        )  # pyright: ignore[reportCallIssue]
