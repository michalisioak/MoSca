from dataclasses import dataclass
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from lib_moca.camera import MonocularCameras
from lib_prior.prior_loading import Saved2D
from gsplat import rasterization
from scene.losses.dep import compute_dep_loss
from scene.losses.rgb import compute_rgb_loss
from scene.losses.ssim import compute_ssim_loss
from scene.static import StaticScene
import os.path as osp
import os


@dataclass
class PhotometricFitConfig:
    total_steps = 8000
    topo_update_feq = 50
    skinning_corr_start_steps = 1e10
    s_gs_ctrl_start_ratio = 0.2
    s_gs_ctrl_end_ratio = 0.9
    d_gs_ctrl_start_ratio = 0.2
    d_gs_ctrl_end_ratio = 0.9
    lambda_ssim = 0.1
    lambda_rgb = 1.0
    lambda_dep = 1.0


def static_photometric_fit(
    ws: str,
    s2d: Saved2D,
    cams: MonocularCameras,
    s_model: StaticScene,
    cfg: PhotometricFitConfig,
    dep_st_invariant=True,
):
    logger.info("Starting Warmup Fit...")
    warmup_ws = osp.join(ws, "warmup")
    os.makedirs(warmup_ws, exist_ok=True)
    torch.cuda.empty_cache()
    cam_param_list = cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
    if len(cam_param_list) > 0:
        optimizer_cam = torch.optim.Adam(
            cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
        )
    else:
        optimizer_cam = None

    loss_rgb_list = []
    loss_dep_list = []

    s_gs_ctrl_start = int(cfg.total_steps * cfg.s_gs_ctrl_start_ratio)
    d_gs_ctrl_start = int(cfg.total_steps * cfg.d_gs_ctrl_start_ratio)
    assert s_gs_ctrl_start >= 0
    assert d_gs_ctrl_start >= 0

    base_u, base_v = np.meshgrid(np.arange(s2d.W), np.arange(s2d.H))
    base_uv = np.stack([base_u, base_v], -1)
    base_uv = torch.tensor(base_uv, device=s2d.rgb.device).long()

    for step in tqdm(range(cfg.total_steps)):
        if optimizer_cam is not None:
            optimizer_cam.zero_grad()
        cams.zero_grad()
        s2d.zero_grad()

        view_ind = np.random.choice(cams.T, 1, replace=False).tolist()

        means, quats, scales, opacities, sph = s_model.forward()
        image, alpha, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            height=s2d.H,
            width=s2d.W,
            Ks=cams.K(s2d.H, s2d.W),
            viewmats=cams.T_cw(view_ind),
        )

        # compute losses
        rgb_sup_mask = s2d.get_mask_by_key(sup_mask_type)[view_ind]
        loss_rgb = compute_rgb_loss(
            s2d.rgb[view_ind].detach().clone(), image, rgb_sup_mask
        )
        dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind]
        loss_dep = compute_dep_loss(
            s2d.dep[view_ind].detach().clone(),
            info["depths"],
            dep_sup_mask,
            st_invariant=dep_st_invariant,
        )
        loss_ssim = compute_ssim_loss(
            s2d.rgb[view_ind].detach().clone(), image, rgb_sup_mask
        )

        loss = (
            loss_rgb * cfg.lambda_rgb
            + loss_dep * cfg.lambda_dep
            + loss_ssim * cfg.lambda_ssim
        )

        s_model.step_pre_backward(step, info)
        loss.backward()
        s_model.step_post_backward(step, info)
        optimizer_cam.step()

        loss_rgb_list.append(loss_rgb.item())
        loss_dep_list.append(loss_dep.item())

    torch.save(s_model.state_dict(), osp.join(warmup_ws, "s_model.pth"))
    torch.save(cams.state_dict(), osp.join(warmup_ws, "cam.pth"))
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
    plt.savefig(osp.join(warmup_ws, "optim_loss.jpg"))
    plt.close()
    torch.cuda.empty_cache()
