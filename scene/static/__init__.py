import torch

from monocular_cameras.cameras import MonocularCameras
from scene.cfg import GaussianSplattingConfig
from scene.fetch import fetch_leaves_in_world_frame, LeavesFetchConfig

from .model import StaticScene


@torch.no_grad()
def create_static_model(
    dep: torch.Tensor,
    rgb: torch.Tensor,
    s_mask: torch.Tensor,
    cams: MonocularCameras,
    fetch_cfg: LeavesFetchConfig,
    gs_cfg: GaussianSplattingConfig,
) -> StaticScene:
    mu_init, q_init, s_init, o_init, rgb_init, _ = fetch_leaves_in_world_frame(
        cams=cams,
        cfg=fetch_cfg,
        input_mask_list=s_mask,
        input_dep_list=dep,
        input_rgb_list=rgb,
    )
    s_model: StaticScene = StaticScene(
        means=mu_init,
        quats=q_init,
        scales=s_init,
        opacities=o_init,
        rgbs=rgb_init,
        cfg=gs_cfg,
    )
    return s_model
