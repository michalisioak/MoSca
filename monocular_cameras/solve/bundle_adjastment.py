import os
import torch
import os.path as osp
from tqdm import tqdm
from loguru import logger
from monocular_cameras.cameras import MonocularCameras, CameraConfig
from monocular_cameras.solve.buffer_utils import prepare_track_homo_dep_rgb_buffers
from prior.depth.save import save_depth_list
from prior.depth.viz import viz_depth_list
from .robust import positive_th_gaussian_decay

from dataclasses import dataclass, field

# from .viz import BundleAdjastmentVizConfig


@dataclass
class StaticBundleAdjastmentConfig:
    steps: int = 2000
    camera: CameraConfig = field(
        default=CameraConfig(lr_q=0.0003, lr_t=0.0003, lr_f=0.0003)
    )
    lr_dep_c: float | None = 0.001
    lr_dep_s: float | None = 0.001
    lambda_flow = 1.0
    lambda_depth = 0.1
    lambda_small_correction = 0.03
    lambda_cam_smooth_trans = 0.0
    lambda_cam_smooth_rot = 0.0
    huber_delta: float | None = None
    max_t_per_step = 10000
    switch_to_ind_step = 500
    max_num_of_tracks = 10000
    depth_correction_after_step = 500
    # viz: BundleAdjastmentVizConfig | None = field(
    #     default_factory=BundleAdjastmentVizConfig
    # )
    save_more_flag: bool = False


def compute_static_ba(
    ws: str,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    s_track: torch.Tensor,
    track_mask: torch.Tensor,
    cfg: StaticBundleAdjastmentConfig,
    cams: MonocularCameras,
    # viz
    # ! all these robust weights are computed on the fly, which assume that the initializaiton is good
    depth_decay_th=2.0,
    depth_decay_sigma=1.0,
    std_decay_th=0.2,
    std_decay_sigma=0.2,
    #
    optimizer_class=torch.optim.Adam,
):
    # prepare dense track
    # s_track = s2d.track[:, s2d.static_track_mask, :2].clone()
    s_track = s_track[..., :2].clone()
    device = s_track.device
    if cfg.max_num_of_tracks < s_track.shape[1]:
        logger.info(
            f"Track is too dense {s_track.shape[1]}, radom sample to {cfg.max_num_of_tracks}"
        )
        choice = torch.randperm(s_track.shape[1])[: cfg.max_num_of_tracks]
        s_track = s_track[:, choice]

    homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
        rgb=rgb, dep=dep, track=s_track, track_mask=track_mask
    )

    # viz = BundleAdjastmentVizualizer(
    #     log_dir=ws,
    #     total_steps=cfg.steps,
    #     cfg=cfg.viz,
    #     rgb=rgb_list.detach().cpu().numpy(),
    #     static_tracks=s_track.cpu().numpy(),
    # )

    # * start solve global init of the camera
    logger.info(
        f"Static Scaffold BA: Depth correction after {cfg.depth_correction_after_step}; Lr Scheduling and Ind after {cfg.switch_to_ind_step} steps (total {cfg.steps})"
    )
    param_scale = torch.ones(cams.T).to(device)
    param_scale.requires_grad_(True)
    param_depth_corr = torch.zeros_like(dep_list).clone()
    param_depth_corr.requires_grad_(True)
    optim_list = cams.get_optimizable_list(cfg.camera)
    if cfg.lr_dep_s is not None and cfg.lr_dep_s > 0:
        optim_list.append(
            {"params": [param_scale], "lr": cfg.lr_dep_s, "name": "cam_scale"}
        )
    if cfg.lr_dep_c is not None and cfg.lr_dep_c > 0:
        optim_list.append(
            {
                "params": [param_depth_corr],
                "lr": cfg.lr_dep_c,
                "name": "depth_correction",
            }
        )
    optimizer = optimizer_class(optim_list)
    scheduler = None
    s_track_valid_mask_w = track_mask.float()
    s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)

    huber_loss = None
    if cfg.huber_delta is not None and cfg.huber_delta > 0:
        logger.info(f"Use Huber Loss with delta={cfg.huber_delta}")
        huber_loss = torch.nn.HuberLoss(reduction="none", delta=cfg.huber_delta)

    logger.info(f"Start Static BA with {cams.T} frames and {dep_list.shape[1]} points")

    for step in tqdm(range(cfg.steps)):
        if step == cfg.switch_to_ind_step:
            logger.info(
                "Switch to Independent Camera Optimization and Start Scheduling"
            )
            cams.disable_delta()
            optim_list = cams.get_optimizable_list(cfg.camera)
            if cfg.lr_dep_s is not None and cfg.lr_dep_s > 0:
                optim_list.append(
                    {"params": [param_scale], "lr": cfg.lr_dep_s, "name": "cam_scale"}
                )
            if cfg.lr_dep_c is not None and cfg.lr_dep_c > 0:
                optim_list.append(
                    {
                        "params": [param_depth_corr],
                        "lr": cfg.lr_dep_c,
                        "name": "dep_correction",
                    }
                )
            optimizer = optimizer_class(optim_list)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                cfg.steps - cfg.switch_to_ind_step,
                eta_min=min(cfg.camera.lr_q or 0, cfg.camera.lr_t or 0) / 10.0,
            )

        optimizer.zero_grad()

        ########################
        dep_scale = param_scale.abs()
        dep_scale = dep_scale / dep_scale.mean()

        scaled_depth_list = dep_list * dep_scale[:, None]
        if step > cfg.depth_correction_after_step:
            scaled_depth_list = scaled_depth_list + param_depth_corr
            dep_corr_loss = abs(param_depth_corr).mean()
        else:
            dep_corr_loss = torch.zeros_like(dep_scale[0])
        point_ref = get_world_points(homo_list, scaled_depth_list, cams)  # T,N,3

        # transform to each frame!
        if cams.T > cfg.max_t_per_step:
            tgt_inds = torch.randperm(cams.T)[: cfg.max_t_per_step].to(device)
        else:
            tgt_inds = torch.arange(cams.T).to(device)
        R_cw, t_cw = cams.Rt_cw_list()
        R_cw, t_cw = R_cw[tgt_inds], t_cw[tgt_inds]

        point_ref_to_every_frame = (
            torch.einsum("tij,snj->stni", R_cw, point_ref) + t_cw[None, :, None]
        )  # Src,Tgt,N,3
        uv_src_to_every_frame = cams.project(
            point_ref_to_every_frame
        )  # Src,Tgt,N,3 # Mich maybey is it 2?

        # * robusitify the loss by down weight some curves
        with torch.no_grad():
            projection_singular_mask = abs(point_ref_to_every_frame[..., -1]) < 1e-5
            # no matter where the src is, it should be mapped to every frame with the gt tracking
            cross_time_mask = (track_mask[:, None] * track_mask[None, tgt_inds]).float()
            cross_time_mask = (
                cross_time_mask * (~projection_singular_mask).float()
            )  # Src,Tgt,N

            depth_robust_w = positive_th_gaussian_decay(
                abs(scaled_depth_list), depth_decay_th, depth_decay_sigma
            )

            point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
            point_ref_std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
            point_ref_std_robust_w = (point_ref_std * s_track_valid_mask_w).sum(0)
            point_ref_std_robust_w = positive_th_gaussian_decay(
                point_ref_std_robust_w, std_decay_th, std_decay_sigma
            )

            robust_w = depth_robust_w * point_ref_std_robust_w[None]
            cross_robust_time_mask = robust_w[:, None] * robust_w[None, tgt_inds]
            cross_time_mask = cross_time_mask * cross_robust_time_mask.detach()

        uv_target = homo_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1, -1
        )
        uv_loss_i = (uv_src_to_every_frame - uv_target).norm(dim=-1)

        if cfg.huber_delta is not None and cfg.huber_delta > 0:
            assert huber_loss is not None
            uv_loss_i = huber_loss(uv_loss_i, torch.zeros_like(uv_loss_i))
        uv_loss = (uv_loss_i * cross_time_mask).sum() / (cross_time_mask.sum() + 1e-6)

        # compute depth loss
        dep_target = scaled_depth_list[None, tgt_inds].expand(
            len(uv_src_to_every_frame), -1, -1
        )
        warped_depth = point_ref_to_every_frame[..., -1]

        dep_consistency_i = 0.5 * abs(
            dep_target / torch.clamp(warped_depth, min=1e-6) - 1
        ) + 0.5 * abs(warped_depth / torch.clamp(dep_target, min=1e-6) - 1)
        # todo: this may be unstable... for fare away depth points!!!
        if cfg.huber_delta is not None and cfg.huber_delta > 0:
            assert huber_loss is not None
            dep_consistency_i = huber_loss(
                dep_consistency_i, torch.zeros_like(dep_consistency_i)
            )
        dep_loss = (dep_consistency_i * cross_time_mask).sum() / (
            cross_time_mask.sum() + 1e-6
        )

        # camera smoothness reg
        if cfg.lambda_cam_smooth_rot > 0 or cfg.lambda_cam_smooth_trans > 0:
            cam_trans_loss, cam_rot_loss = cams.smoothness_loss()
        else:
            cam_trans_loss = torch.zeros_like(dep_loss)
            cam_rot_loss = torch.zeros_like(dep_loss)

        loss: torch.Tensor = (
            cfg.lambda_depth * dep_loss
            + cfg.lambda_flow * uv_loss
            + cfg.lambda_small_correction * dep_corr_loss
            + cfg.lambda_cam_smooth_rot * cam_rot_loss
            + cfg.lambda_cam_smooth_trans * cam_trans_loss
        )

        assert torch.isnan(loss).sum() == 0 and torch.isinf(loss).sum() == 0
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # viz.step(
        #     step=step,
        #     loss=loss,
        #     s_track_valid_mask_w=s_track_valid_mask_w,
        #     point_ref=point_ref,
        #     cam_rot_loss=cam_rot_loss,
        #     dep_corr_loss=dep_corr_loss,
        #     uv_loss=uv_loss,
        #     dep_loss=dep_loss,
        #     cam_trans_loss=cam_trans_loss,
        #     cams=cams,
        #     param_scale=param_scale,
        # )

    # viz.save()

    # update the depth scale
    # dep_scale = param_scale.abs()
    dep_scale = param_scale.abs()
    dep_scale = dep_scale / dep_scale.mean()
    logger.info(f"Update the S2D depth scale with {dep_scale}")
    # s2d.rescale_depth(dep_scale)
    bundle_dir = osp.join(ws, "bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    torch.save(cams.state_dict(), osp.join(bundle_dir, "cams.pth"))
    dep = dep.clone() * dep_scale[:, None, None]
    save_depth_list(
        dep_list=list(dep.cpu().numpy()),
        ws=ws,
        name="bundle",
        # invalid_mask_list=(s_visibility == 0),
    )
    viz_depth_list(
        depths=list(dep.cpu().numpy()),
        ws=ws,
        name="bundle",
    )
    return cams, s_track, param_depth_corr.detach().clone()


def get_world_points(
    homo_list: torch.Tensor,
    dep_list: torch.Tensor,
    cams: MonocularCameras,
    cam_t_list=None,
):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = cams.backproject(homo_list.reshape(-1, 2), dep_list.reshape(-1))
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]
    return point_world
