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
        N = rgbs.shape[0]
        self.cfg = cfg

        # gsplat expects: [N, (sph_order + 1)², 3]
        num_coeffs = (cfg.sph_order + 1) ** 2
        shs = torch.zeros(N, num_coeffs, 3, device=self.device)
        shs[:, 0, :] = rgbs
        if cfg.sph_order > 0:
            std = 0.01
            shs[:, 1:, :] = torch.randn(N, num_coeffs - 1, 3, device=self.device) * std

        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(
                    torch.ones(means.shape[0]) if opacities is None else opacities
                ),
                "quats": nn.Parameter(
                    torch.ones((means.shape[0], 4)) if quats is None else quats
                ),
                "shs": nn.Parameter(shs),
            }
        )

        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            name: cfg.get_optimizer()(
                params=param,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            for name, param in self.params.items()
        }
        self.strategy = cfg.strategy
        self.strategy.check_sanity(self.params, self.optimizers)
        self.state = self.strategy.initialize_state()
        self.summary()

    def summary(self):
        logger.info(f"StaticScene: {len(self.xyz) / 1000.0:.1f}K points")

    def forward(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.params["means"],
            self.params["quats"],
            self.params["scales"],
            self.params["opacities"],
            self.params["shs"],
        )

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
        )  # type: ignore
