from dataclasses import dataclass, field
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monocular_cameras.cameras import MonocularCameras, CameraConfig
from gsplat import rasterization
from scene.losses.dep import compute_dep_loss
from scene.losses.rgb import compute_rgb_loss
from scene.losses.ssim import compute_ssim_loss
from scene.static.model import StaticScene
from scene.dynamic.model import DynamicScene
import os.path as osp
import os


@dataclass
class PhotometricFitConfig:
    total_steps = 10_000
    topo_update_feq = 50
    skinning_corr_start_steps = 1e10
    s_gs_ctrl_start_ratio = 0.2
    s_gs_ctrl_end_ratio = 0.9
    d_gs_ctrl_start_ratio = 0.2
    d_gs_ctrl_end_ratio = 0.9
    lambda_ssim = 0.1
    lambda_rgb = 1.0
    lambda_dep = 1.0
    camera: CameraConfig = field(default_factory=CameraConfig)


def photometric_fit(
    ws: str,
    dep: torch.Tensor,
    rgb: torch.Tensor,
    cams: MonocularCameras,
    s_model: StaticScene,
    d_model: DynamicScene,
    cfg: PhotometricFitConfig,
    dep_st_invariant=True,
):
    logger.info("Starting photometric Fit...")
    width, height = rgb.shape[2], rgb.shape[1]
    fit_dir = osp.join(ws, "photometric")
    os.makedirs(fit_dir, exist_ok=True)
    torch.cuda.empty_cache()
    optimizer_cam = cams.get_optimizer(cfg.camera)

    loss_rgb_list = []
    loss_dep_list = []

    s_gs_ctrl_start = int(cfg.total_steps * cfg.s_gs_ctrl_start_ratio)
    d_gs_ctrl_start = int(cfg.total_steps * cfg.d_gs_ctrl_start_ratio)
    assert s_gs_ctrl_start >= 0
    assert d_gs_ctrl_start >= 0

    base_u, base_v = np.meshgrid(np.arange(width), np.arange(height))
    base_uv = np.stack([base_u, base_v], -1)
    base_uv = torch.tensor(base_uv, device=rgb.device).long()

    for step in tqdm(range(cfg.total_steps)):
        if optimizer_cam is not None:
            optimizer_cam.zero_grad()
        cams.zero_grad()

        view_ind = np.random.choice(cams.T, 1, replace=False).tolist()

        s_means, s_quats, s_scales, s_opacities, s_shs = s_model.forward()
        d_means, d_quats, d_scales, d_opacities, d_shs = d_model.forward(view_ind)
        image, alpha, info = rasterization(
            means=torch.cat((s_means, d_means)),
            quats=torch.cat((s_quats, d_quats)),
            scales=torch.cat((s_scales, d_scales)),
            opacities=torch.cat((s_opacities, d_opacities)),
            colors=torch.cat((s_shs, d_shs)),
            height=height,
            width=width,
            Ks=cams.K(height, width),
            viewmats=cams.T_cw(view_ind),
        )

        # compute losses
        loss_rgb = compute_rgb_loss(
            rgb[view_ind].detach().clone(),
            image,
        )
        loss_dep = compute_dep_loss(
            dep[view_ind].detach().clone(),
            info["depths"],
            st_invariant=dep_st_invariant,
        )
        loss_ssim = compute_ssim_loss(
            rgb[view_ind].detach().clone(),
            image,
        )

        loss = (
            loss_rgb * cfg.lambda_rgb
            + loss_dep * cfg.lambda_dep
            + loss_ssim * cfg.lambda_ssim
        )

        s_model.step_pre_backward(step, info)
        d_model.step_pre_backward(step, info)
        loss.backward()
        s_model.step_post_backward(step, info)
        d_model.step_post_backward(step, info)
        if optimizer_cam is not None:
            optimizer_cam.step()

        loss_rgb_list.append(loss_rgb.item())
        loss_dep_list.append(loss_dep.item())

    torch.save(s_model.state_dict(), osp.join(fit_dir, "s_model.pth"))
    torch.save(d_model.state_dict(), osp.join(fit_dir, "d_model.pth"))
    torch.save(cams.state_dict(), osp.join(fit_dir, "cam.pth"))
    # viz
    plt.figure(figsize=(30, 8))
    for plt_i, plt_pack in enumerate(
        [
            ("loss_rgb", loss_rgb_list),
            ("loss_dep", loss_dep_list),
        ]
    ):
        plt.subplot(2, 10, plt_i + 1)
        plt.plot(plt_pack[1])
        plt.title(plt_pack[0] + f" End={plt_pack[1][-1]:.6f}")
    plt.savefig(osp.join(fit_dir, "optim_loss.jpg"))
    plt.close()
    torch.cuda.empty_cache()
