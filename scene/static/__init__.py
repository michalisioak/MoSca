import torch


from monocular_cameras.cameras import MonocularCameras
from scene.cfg import GaussianSplattingConfig
from scene.fetch import fetch_leaves_in_world_frame, LeavesFetchConfig

from .model import StaticScene


@torch.no_grad()
def create_static_model(
    dep: torch.Tensor,
    rgb: torch.Tensor,
    gather_mask: torch.Tensor,
    cams: MonocularCameras,
    fetch_cfg: LeavesFetchConfig,
    gs_cfg: GaussianSplattingConfig,
    radius_init_factor: float,
    opacity_init_factor: float,
) -> StaticScene:
    mu_init, q_init, s_init, o_init, rgb_init, _ = fetch_leaves_in_world_frame(
        cams=cams,
        cfg=fetch_cfg,
        gather_mask=gather_mask,
        dep=dep,
        rgb=rgb,
    )
    s_model: StaticScene = StaticScene(
        means=mu_init,
        quats=q_init,
        scales=s_init * radius_init_factor,
        opacities=o_init * opacity_init_factor,
        rgbs=rgb_init,
        cfg=gs_cfg,
    )
    s_model.summary()
    return s_model
