from typing import Dict, Mapping, Tuple
import torch
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from scene.cfg import GaussianSplattingConfig


def static_scene_init(
    means: torch.Tensor,  # N,3
    rgbs: torch.Tensor,  # N,3
    scales: torch.Tensor,  # N,3
    quats: torch.Tensor,  # N,4
    opacities: torch.Tensor,  # N,1
    ids: torch.Tensor | None = None,
    max_scale: float = 0.1,  # use sigmoid activation, can't be too large
    min_scale: float = 0.0,
    max_sph_order: int = 0,
    cfg: GaussianSplattingConfig = GaussianSplattingConfig(),
) -> Tuple[torch.nn.ParameterDict, Mapping[str, torch.optim.Optimizer]]:
    params = [
        # name, value, lr
        ("static_means", torch.nn.Parameter(means), cfg.means_lr),
        ("static_scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("static_quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("static_opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
    ]
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to("cuda")
    optimizer_class = cfg.get_optimizer()
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers
