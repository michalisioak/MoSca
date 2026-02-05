import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import logging
import imageio
from prior.loading.saved_prior import Saved2D
from matplotlib import cm
from torch_geometric.nn import knn
from monocular_cameras import MonocularCameras, CameraConfig
from .robust import positive_th_gaussian_decay

from dataclasses import dataclass, field

from .viz import BundleAdjastmentVizConfig, BundleAdjastmentVizualizer


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
    viz: BundleAdjastmentVizConfig | None = field(
        default_factory=BundleAdjastmentVizConfig
    )
    save_more_flag: bool = False


def compute_static_ba(
    ws: str,
    s_track,
    s_track_valid_mask,
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
    s_track_valid_mask = s_track_valid_mask.clone()
    device = s_track.device
    if cfg.max_num_of_tracks < s_track.shape[1]:
        logging.info(
            f"Track is too dense {s_track.shape[1]}, radom sample to {cfg.max_num_of_tracks}"
        )
        choice = torch.randperm(s_track.shape[1])[: cfg.max_num_of_tracks]
        s_track = s_track[:, choice]
        s_track_valid_mask = s_track_valid_mask[:, choice]

    homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
        s2d, s_track, s_track_valid_mask, torch.arange(s2d.T).to(device)
    )

    viz = BundleAdjastmentVizualizer(
        log_dir=ws,
        total_steps=cfg.steps,
        cfg=cfg.viz,
        rgb=rgb_list.detach().cpu().numpy(),
        static_tracks=s_track.cpu().numpy(),
        static_track_valid_mask=s_track_valid_mask.cpu().numpy(),
    )

    # * start solve global init of the camera
    logging.info(
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
    s_track_valid_mask_w = s_track_valid_mask.float()
    s_track_valid_mask_w = s_track_valid_mask_w / s_track_valid_mask_w.sum(0)

    huber_loss = None
    if cfg.huber_delta is not None and cfg.huber_delta > 0:
        logging.info(f"Use Huber Loss with delta={cfg.huber_delta}")
        huber_loss = torch.nn.HuberLoss(reduction="none", delta=cfg.huber_delta)

    logging.info(f"Start Static BA with {cams.T} frames and {dep_list.shape[1]} points")

    for step in tqdm(range(cfg.steps)):
        if step == cfg.switch_to_ind_step:
            logging.info(
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
            cross_time_mask = (
                s_track_valid_mask[:, None] * s_track_valid_mask[None, tgt_inds]
            ).float()
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

        viz.step(
            step=step,
            loss=loss,
            s_track_valid_mask_w=s_track_valid_mask_w,
            point_ref=point_ref,
            cam_rot_loss=cam_rot_loss,
            dep_corr_loss=dep_corr_loss,
            uv_loss=uv_loss,
            dep_loss=dep_loss,
            cam_trans_loss=cam_trans_loss,
            cams=cams,
            param_scale=param_scale,
        )

    viz.save()

    # update the depth scale
    # dep_scale = param_scale.abs()
    dep_scale = param_scale.abs()
    dep_scale = dep_scale / dep_scale.mean()
    logging.info(f"Update the S2D depth scale with {dep_scale}")
    s2d.rescale_depth(dep_scale)
    torch.save(cams.state_dict(), osp.join(ws, "bundle_cams.pth"))
    torch.save(
        {
            # sol
            "dep_scale": dep_scale,  # ! important, to later rescale the depth
            "dep_correction": param_depth_corr,
            "s_track": s_track,
            "s_track_mask": s_track_valid_mask,
        },
        osp.join(ws, "bundle.pth"),
    )
    # also save a reconstructed point cloud
    # if cfg.save_more_flag:
    #     np.savetxt(
    #         osp.join(log_dir, "static_scaffold_pcl_unmerged.xyz"),
    #         # torch.cat([point_ref, rgb_list], -1).reshape(-1, 6).detach().cpu().numpy(),
    #         fmt="%.6f",
    #     )
    return cams, s_track, s_track_valid_mask, param_depth_corr.detach().clone()


@torch.no_grad()
def deform_depth_map(
    depth_list,
    mask_list,
    cams: MonocularCameras,
    track_uv_list,
    track_mask_list,
    dep_correction,
    K=16,
    rbf_factor=0.333,
    viz_fn=None,
):
    # depth_list: T,H,W; track_uv_list: T,N,2; track_mask_list: T,N; src_buffer: T,N,C
    logging.info("Deforming depth")
    T = len(track_mask_list)
    H, W = depth_list[0].shape
    assert depth_list.shape == mask_list.shape
    assert T == len(track_uv_list) == len(dep_correction)
    assert T == len(depth_list) == len(mask_list)
    homo_map = torch.from_numpy(get_homo_coordinate_map(H, W)).to(depth_list[0])

    dep_corr_map_list, dep_new_map_list = [], []
    for tid in tqdm(range(T)):
        mask2d = mask_list[tid]
        scf_mask = track_mask_list[tid]
        dep_map = depth_list[tid]
        scf_uv = track_uv_list[tid][scf_mask]
        scf_int_uv, scf_inside_mask = round_int_coordinates(scf_uv, H, W)
        if not scf_inside_mask.all():
            logging.warning(
                f"Warning, {(~scf_inside_mask).sum()} invalid uv in t={tid}! may due to round accuracy"
            )

        scf_dep = query_image_buffer_by_pix_int_coord(depth_list[tid], scf_int_uv)
        scf_homo = query_image_buffer_by_pix_int_coord(homo_map, scf_int_uv)
        # this pts is used to distribute the carrying interp_src in 3D cam frame
        scf_cam_pts = cams.backproject(scf_homo, scf_dep)
        dst_cam_pts = cams.backproject(homo_map[mask2d], dep_map[mask2d])
        scf_buffer = dep_correction[tid][scf_mask]

        interp_dep_corr = spatial_interpolation(
            src_xyz=scf_cam_pts,
            src_buffer=scf_buffer[:, None],
            query_xyz=dst_cam_pts,
            K=K,
            rbf_sigma_factor=rbf_factor,
        )

        # viz
        dep_corr_map = torch.zeros_like(dep_map)
        dep_corr_map[mask2d] = interp_dep_corr.squeeze(-1)
        # scf_corr_interp = query_image_buffer_by_pix_int_coord(dep_corr_map, scf_int_uv)
        # check_interp_error = (
        #     abs(scf_corr_interp - scf_buffer.squeeze(-1)).median()
        #     / abs(scf_buffer).median()
        # )s
        dep_corr_map_list.append(dep_corr_map.detach())
        dep_new_map_list.append((dep_corr_map + dep_map).detach())

    if viz_fn is not None:
        viz_corr_list, viz_dep_list = [], []
        for tid in tqdm(range(T)):
            viz_corr = dep_corr_map_list[tid]
            viz_dep = dep_new_map_list[tid]
            viz_corr_radius = abs(viz_corr).max()
            viz_corr = (viz_corr / viz_corr_radius) + 0.5
            viz_dep = (viz_dep - viz_dep.min()) / (viz_dep.max() - viz_dep.min())
            viz_corr = cm.get_cmap("viridis")(viz_corr.cpu().numpy())
            viz_dep = cm.get_cmap("viridis")(viz_dep.cpu().numpy())
            viz_corr = (viz_corr * 255).astype(np.uint8)
            viz_dep = (viz_dep * 255).astype(np.uint8)
            viz_corr_list.append(viz_corr)
            viz_dep_list.append(viz_dep)
        imageio.mimsave(viz_fn.replace(".mp4", "_corr.mp4"), viz_corr_list)
        imageio.mimsave(viz_fn.replace(".mp4", "_dep_corr.mp4"), viz_dep_list)

    dep_new_map_list = torch.stack(dep_new_map_list, 0)
    return dep_new_map_list


# OLD
# def spatial_interpolation(src_xyz, src_buffer, query_xyz, K=16, rbf_sigma_factor=0.333):
#     # src_xyz: M,3 src_buffer: M,C query_xyz: N,3
#     # build RBG on each src and smoothly interpolate the buffer to query
#     # first construct src_xyz nn graph
#     _dist_sq_to_nn, _, _ = knn_points(src_xyz[None], src_xyz[None], K=2)
#     dist_to_nn = torch.sqrt(torch.clamp(_dist_sq_to_nn[0, :, 1:], min=1e-8)).squeeze(-1)
#     rbf_sigma = dist_to_nn * rbf_sigma_factor  # M
#     # find the nearest K neighbors for each query point to the src
#     dist_sq, ind, _ = knn_points(query_xyz[None], src_xyz[None], K=K)
#     dist_sq, ind = dist_sq[0], ind[0]

#     w = torch.exp(-dist_sq / (2.0 * (rbf_sigma[ind] ** 2)))  # N,K
#     w = w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)

#     value = src_buffer[ind]  # N,K,C
#     ret = torch.einsum("nk, nkc->nc", w, value)
#     return ret


def get_homo_coordinate_map(H, W):
    # the grid take the short side has (-1,+1)
    if H > W:
        u_range = [-1.0, 1.0]
        v_range = [-float(H) / W, float(H) / W]
    else:  # H<=W
        u_range = [-float(W) / H, float(W) / H]
        v_range = [-1.0, 1.0]
    # make uv coordinate
    u, v = np.meshgrid(
        np.linspace(u_range[0], u_range[1], W), np.linspace(v_range[0], v_range[1], H)
    )
    uv = np.stack([u, v], axis=-1)  # H,W,2
    return uv


def round_int_coordinates(coord, H, W):
    ret = coord.round().long()
    valid_mask = (
        (ret[..., 0] >= 0) & (ret[..., 0] < W) & (ret[..., 1] >= 0) & (ret[..., 1] < H)
    )
    ret[..., 0] = torch.clamp(ret[..., 0], 0, W - 1)
    ret[..., 1] = torch.clamp(ret[..., 1], 0, H - 1)
    return ret, valid_mask


def query_image_buffer_by_pix_int_coord(buffer: torch.Tensor, pixel_int_coordinate):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    # if isinstance(ret, np.ndarray):
    #     ret = ret.copy()
    return ret


def prepare_track_homo_dep_rgb_buffers(
    dep:, track: torch.Tensor, track_mask: torch.Tensor, t_list
):
    # track: T,N,2, track_mask: T,N
    device = track.device
    homo_list, ori_dep_list, rgb_list = [], [], []
    for ind, tid in enumerate(t_list):
        _uv = track[ind]
        _int_uv, _inside_mask = round_int_coordinates(_uv, s2d.H, s2d.W)
        _dep = query_image_buffer_by_pix_int_coord(
            dep[tid].clone().to(device), _int_uv
        )
        _homo = query_image_buffer_by_pix_int_coord(
            homo_map.clone().to(device), _int_uv
        )
        ori_dep_list.append(_dep.to(device))
        homo_list.append(_homo.to(device))
        # for viz purpose
        _rgb = query_image_buffer_by_pix_int_coord(
            rgb[tid].clone().to(device), _int_uv
        )
        rgb_list.append(_rgb.to(device))
    rgb_list = torch.stack(rgb_list, 0)
    ori_dep_list = torch.stack(ori_dep_list, 0)
    homo_list = torch.stack(homo_list)
    ori_dep_list[~track_mask] = -1
    homo_list[~track_mask] = 0.0
    return homo_list, ori_dep_list, rgb_list


def query_buffers_by_track(image_buffer, track, track_mask, default_value=0.0):
    # image_buffer: T,H,W,C; track: T,N,2, track_mask: T,N
    assert image_buffer.ndim == 4 and track.ndim == 3 and track_mask.ndim == 2
    assert len(image_buffer) == len(track) == len(track_mask)
    T, H, W, C = image_buffer.shape
    N = track.shape[1]
    ret_buffer = torch.ones(T, N, C).to(image_buffer) * default_value

    for i in range(T):
        _uv = track[i][..., :2]
        _int_uv, _inside_mask = round_int_coordinates(_uv, H, W)
        _value = query_image_buffer_by_pix_int_coord(image_buffer[i].clone(), _int_uv)
        valid_mask = track_mask[i] & _inside_mask
        # for outside, put default value
        _value[~valid_mask] = default_value
        ret_buffer[i] = _value
    return ret_buffer


def get_world_points(homo_list, dep_list, cams, cam_t_list=None):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = cams.backproject(homo_list.reshape(-1, 2), dep_list.reshape(-1))
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]
    return point_world


def fovdeg2focal(fov_deg):
    focal = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    return focal


def track2undistroed_homo(track, H, W):
    # the short side is -1,1, the long side may exceed
    H, W = float(H), float(W)
    L = min(H, W)
    u, v = track[..., 0], track[..., 1]
    u = 2.0 * u / L - W / L
    v = 2.0 * v / L - H / L
    uv = torch.stack([u, v], -1)
    return uv


def spatial_interpolation(
    src_xyz: torch.Tensor,
    src_buffer: torch.Tensor,
    query_xyz: torch.Tensor,
    K: int = 16,
    rbf_sigma_factor: float = 0.333,
):
    M = src_xyz.shape[0]
    src_indices = knn(src_xyz, src_xyz, k=2, batch_x=None, batch_y=None)
    row, col = src_indices
    dist_vec = src_xyz[row] - src_xyz[col]
    dist_sq_all = torch.sum(dist_vec.pow(2), dim=1)
    dist_sq_reshaped = dist_sq_all.view(M, 2)
    dist_sq_nn = dist_sq_reshaped[:, 1]
    dist_to_nn = torch.sqrt(torch.clamp(dist_sq_nn, min=1e-8))
    rbf_sigma = dist_to_nn * rbf_sigma_factor  # (M,)
    query_indices = knn(src_xyz, query_xyz, k=K, batch_x=None, batch_y=None)
    query_row, src_col = query_indices
    N = query_xyz.shape[0]
    ind = src_col.view(N, K)  # (N, K)
    query_xyz_expanded = query_xyz.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
    src_xyz_neighbors = src_xyz[ind]  # (N, K, 3)
    dist_sq = torch.sum(
        (query_xyz_expanded - src_xyz_neighbors).pow(2), dim=2
    )  # (N, K)
    rbf_sigma_neighbors = rbf_sigma[ind]  # (N, K)
    w = torch.exp(-dist_sq / (2.0 * (rbf_sigma_neighbors**2)))  # (N, K)
    w = w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)  # Normalize
    value = src_buffer[ind]  # (N, K, C)
    ret = torch.einsum("nk, nkc->nc", w, value)
    return ret
