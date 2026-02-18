# functions for initialize the missing scaffold
from dataclasses import dataclass, field
from loguru import logger
import torch
import logging
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d


from motion_scaffold.losses.arap import compute_arap_loss
from motion_scaffold.losses.physical import compute_velocity_acceleration_loss
from . import MoSca


@torch.no_grad()
def slot_o3d_outlier_identifyication(
    curve_xyz: torch.Tensor, curve_mask: torch.Tensor, nb_neighbors=20, std_ratio=2.0
):
    # curve_xyz: T,N,3, tensor

    assert curve_xyz.ndim == 3
    T, N, _ = curve_xyz.shape
    ret_inlier_mask = np.ones((T, N)) < 0  # all false
    for t in tqdm(range(T)):
        if not curve_mask[t].any():
            continue
        fg_mask = curve_mask[t].cpu()
        fg_xyz = curve_xyz[t].cpu().numpy()[fg_mask]
        inlier_mask_buffer = np.zeros(len(fg_xyz)) > 0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fg_xyz)

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        inlier_ind = np.asarray(ind)
        if len(inlier_ind) > 0:
            inlier_mask_buffer[inlier_ind] = True  # len(fg_xyz)
        _t_inlier_mask = ret_inlier_mask[t]
        _t_inlier_mask[fg_mask] = inlier_mask_buffer
        ret_inlier_mask[t] = _t_inlier_mask
    ret_inlier_mask = torch.from_numpy(ret_inlier_mask).bool().to(curve_xyz.device)
    logging.warning(
        f"O3D outlier has {ret_inlier_mask.sum() / curve_mask.sum() * 100:.2f}% inliers ({ret_inlier_mask.sum()} inliers out of {curve_mask.sum()})"
    )
    return ret_inlier_mask


@torch.no_grad()
def curve_shaking_identification(curve_xyz, shacking_th=0.2):
    # the queried curve
    # T,N,3
    ref = (curve_xyz[2:] + curve_xyz[:-2]) / 2.0
    ref = torch.cat([curve_xyz[1:2], ref, curve_xyz[-2:-1]], 0)
    diff = (curve_xyz - ref).norm(dim=-1)
    inlier = diff < shacking_th
    return inlier


def __compute_physical_losses__(
    scf: MoSca,
    temporal_diff_shift: list[int],
    temporal_diff_weight: list[float],
    max_time_window: int,
    reduce="sum",
    square=False,
):
    if scf.T > max_time_window:
        start = torch.randint(0, scf.T - max_time_window + 1, (1,)).item()
        sup_tids = torch.arange(start, start + max_time_window)
    else:
        sup_tids = torch.arange(scf.T)
    sup_tids = sup_tids.to(scf.device)

    # * compute losses from the scaffold
    loss_coord, loss_len = compute_arap_loss(
        scf,
        tids=sup_tids,
        temporal_diff_shift=temporal_diff_shift,
        temporal_diff_weight=temporal_diff_weight,
        reduce_type=reduce,
        square=square,
    )
    loss_p_vel, loss_q_vel, loss_p_acc, loss_q_acc = compute_velocity_acceleration_loss(
        scf=scf, time_indices=sup_tids, reduce_type=reduce, square=square
    )
    return loss_coord, loss_len, loss_p_acc, loss_q_acc, loss_p_vel, loss_q_vel


@dataclass
class ScaffoldFitConfig:
    lr_q = 0.1
    lr_p = 0.1
    lr_sig = 0.03
    total_steps = 1000
    max_time_window = 200
    temporal_diff_shift = [1]
    temporal_diff_shift = [1]
    temporal_diff_weight = [1.0]
    lambda_local_coord = 1.0
    lambda_metric_len = 0.0
    lambda_xyz_acc = 0.0
    lambda_q_acc = 0.1
    lambda_xyz_vel = 0.0
    lambda_q_vel = 0.0
    mlevel_resample_steps = 32
    lambda_small_corr = 0.0
    hard_fix_valid = True
    update_full_topo = False
    use_mask_topo = True
    update_all_topo_steps: list[int] = field(default_factory=lambda: [])
    decay_start = 400
    decay_factor = 10.0


def scaffold_fit(
    ws,
    scf: MoSca,
    cfg: ScaffoldFitConfig,
):
    torch.cuda.empty_cache()
    viz_dir = osp.join(ws, "viz", "mosca")
    os.makedirs(viz_dir, exist_ok=True)
    log_dir = osp.join(ws, "mosca")
    os.makedirs(log_dir, exist_ok=True)

    # * The small change is resp. to the init stage
    solid_mask = scf.node_certain
    solid_xyz = scf.node_xyz.clone().detach()

    optimizer = torch.optim.Adam(
        scf.get_optimizable_list(lr_np=cfg.lr_p, lr_nq=cfg.lr_q, lr_nsig=cfg.lr_sig)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        (cfg.total_steps - cfg.decay_start),
        eta_min=min(cfg.lr_p, cfg.lr_q) / cfg.decay_factor,
    )

    loss_list, loss_coord_list, loss_len_list = [], [], []
    loss_small_corr_list = []
    loss_p_acc_list, loss_q_acc_list = [], []
    loss_p_vel_list, loss_q_vel_list = [], []
    loss_flow_xyz_list, loss_flow_nrm_list = [], []
    metric_flow_error_list, metric_normal_angle_list = [], []

    loss_drag_xyz_list, metric_drag_xyz_list = [], []
    loss_dense_flow_xyz_list, loss_dense_flow_nrm_list = [], []
    metric_dense_flow_error_list, metric_dense_normal_angle_list = [], []

    loss_sk_w_consist_list, metric_sk_w_consist_list = [], []
    loss_dense_sk_w_consist_list, metric_dense_sk_w_consist_list = [], []

    # before start, update topo
    scf.update_topology(curve_mask=solid_mask if cfg.use_mask_topo else None)

    logger.info("4DSCF-Solver-loop summary: ")
    logger.info(
        f"total_steps={cfg.total_steps}, decay_start={cfg.decay_start}, hard_fix_valid_flag={cfg.hard_fix_valid}"
    )
    logger.info(f"lr_p={cfg.lr_p}, lr_q={cfg.lr_q}, lr_sig={cfg.lr_sig}")

    for step in tqdm(range(cfg.total_steps)):
        if step in cfg.update_all_topo_steps:
            logger.info(
                f"As specified, update full topo at step={step} without any mask"
            )
            scf.update_topology(curve_mask=None, verbose=True)
        elif step % cfg.mlevel_resample_steps == 0 and step > 0:
            if cfg.update_full_topo:
                scf.update_topology(
                    curve_mask=solid_mask if cfg.use_mask_topo else None, verbose=True
                )
            else:
                scf.update_multilevel_arap_topo(verbose=True)

        optimizer.zero_grad()

        loss_coord, loss_len, loss_p_acc, loss_q_acc, loss_p_vel, loss_q_vel = (
            __compute_physical_losses__(
                scf,
                cfg.temporal_diff_shift,
                cfg.temporal_diff_weight,
                cfg.max_time_window,
            )
        )

        # loss of near original curve
        diff_to_solid_xyz = (scf.node_xyz - solid_xyz.detach()).norm(dim=-1) ** 2
        loss_small_corr = diff_to_solid_xyz[solid_mask].sum()  # ! use sum

        loss: torch.Tensor = (
            cfg.lambda_local_coord * loss_coord
            + cfg.lambda_metric_len * loss_len
            + cfg.lambda_xyz_acc * loss_p_acc
            + cfg.lambda_q_acc * loss_q_acc
            + cfg.lambda_small_corr * loss_small_corr
            + cfg.lambda_xyz_vel * loss_p_vel
            + cfg.lambda_q_vel * loss_q_vel
        )
        with torch.no_grad():
            loss_list.append(loss.item())
            loss_coord_list.append(loss_coord.item())
            loss_len_list.append(loss_len.item())
            loss_p_acc_list.append(loss_p_acc.item())
            loss_q_acc_list.append(loss_q_acc.item())
            loss_small_corr_list.append(loss_small_corr.item())
            loss_p_vel_list.append(loss_p_vel.item())
            loss_q_vel_list.append(loss_q_vel.item())

        loss.backward()
        if cfg.hard_fix_valid:
            scf.mask_xyz_grad(~solid_mask)
        optimizer.step()

        # * control
        if step > cfg.decay_start:
            scheduler.step()

        if step % 50 == 0:
            logging.info(f"step={step}, loss={loss:.6f}")
            msg = "loss_coord={loss_coord:.6f}, loss_len={loss_len:.6f}, loss_p_vel={loss_p_vel:.6f}, loss_R_vel={loss_q_vel:.6f}, loss_p_acc={loss_p_acc:.6f}, loss_R_acc={loss_q_acc:.6f}, loss_small_corr={loss_small_corr:.6f}"
            logging.info(msg)

    plt.figure(figsize=(25, 7))
    for plt_i, plt_pack in enumerate(
        [
            ("loss", loss_list),
            ("loss_coord", loss_coord_list),
            ("loss_len", loss_len_list),
            ("loss_p_acc", loss_p_acc_list),
            ("loss_q_acc", loss_q_acc_list),
            ("loss_p_vel", loss_p_vel_list),
            ("loss_q_vel", loss_q_vel_list),
            ("loss_small_corr", loss_small_corr_list),
            #
            ("loss_flow_xyz", loss_flow_xyz_list),
            ("loss_flow_nrm", loss_flow_nrm_list),
            ("metric_flow_error", metric_flow_error_list),
            ("metric_normal_angle", metric_normal_angle_list),
            #
            ("loss_drag_xyz", loss_drag_xyz_list),
            ("metric_drag_xyz", metric_drag_xyz_list),
            #
            ("loss_dense_flow_xyz", loss_dense_flow_xyz_list),
            ("loss_dense_flow_nrm", loss_dense_flow_nrm_list),
            ("metric_dense_flow_error", metric_dense_flow_error_list),
            ("metric_dense_normal_angle", metric_dense_normal_angle_list),
            #
            ("loss_sk_w_consist", loss_sk_w_consist_list),
            ("metric_sk_w_consist", metric_sk_w_consist_list),
            #
            ("loss_dense_sk_w_consist", loss_dense_sk_w_consist_list),
            ("metric_dense_sk_w_consist", metric_dense_sk_w_consist_list),
        ]
    ):
        plt.subplot(2, 11, plt_i + 1)
        plt.plot(plt_pack[1])
        plt.title(plt_pack[0])
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(osp.join(viz_dir, "dynamic_scaffold_init.jpg"))
    plt.close()

    return scf
