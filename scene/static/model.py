from typing import Any, Dict
from loguru import logger
import torch
from torch import nn
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
        N = rgbs.shape[0]

        # gsplat expects: [N, (sph_order + 1)², 3]
        num_coeffs = (cfg.sph_order + 1) ** 2
        shs = torch.zeros(N, num_coeffs, 3, device=self.device)
        shs[:, 0, :] = rgbs
        if cfg.sph_order > 0:
            std = 0.01
            shs[:, 1:, :] = torch.randn(N, num_coeffs - 1, 3, device=self.device) * std
        self.shs = nn.Parameter(shs)
        optimizer_class = cfg.get_optimizer()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "means": optimizer_class(
                params=self.means,
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
            "sphs": optimizer_class(
                params=self.shs,
                betas=(0.9, 0.999),
                eps=1e-8,
            ),
        }
        self.summary()

    def summary(self):
        logger.info(f"StaticScene: {len(self.xyz) / 1000.0:.1f}K points")

    def forward(self):
        return self.means, self.quats, self.scales, self.opacities, self.shs

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
