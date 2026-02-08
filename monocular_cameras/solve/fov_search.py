from dataclasses import dataclass
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import os.path as osp

from monocular_cameras.backproject import backproject
from monocular_cameras.project import project
from utils3d.torch import fov_to_focal

from monocular_cameras.solve.robust import positive_th_gaussian_decay
from prior.tracking.epi import analyze_track_epi

from lib_moca.intrinsic_helpers import compute_graph_energy as compute_graph_energy_old


@dataclass
class FovSearchConfig:
    fallback_fov: float = 40.0
    search_N: int = 200
    search_start: float = 30.0
    search_end: float = 100.0
    min_valid_covalid = 64
    search_intervals = [10, 20]


@torch.no_grad()
def find_initinsic(
    time: int,
    height: int,
    width: int,
    track: torch.Tensor,
    track_mask: torch.Tensor,
    dep_list,
    depth_decay_th: float,
    depth_decay_sigma: float,
    cfg: FovSearchConfig,
    ws: str,
):
    fov_jump_pair_list = make_pair_list(
        time,
        interval=cfg.search_intervals,
        dense_flag=True,
        track_mask=track_mask,
        min_valid_num=cfg.min_valid_covalid,
    )
    assert len(fov_jump_pair_list) > 0, "no valid pair for FOV search"
    logger.info(f"Start analyzing {len(fov_jump_pair_list)} pairs for FOV search")

    _, _, inlier_list_jumped = analyze_track_epi(
        fov_jump_pair_list, track, track_mask, height=height, width=width
    )
    checked_pair, checked_inlier = [], []
    for pair, inlier in zip(fov_jump_pair_list, inlier_list_jumped):
        if inlier.sum() > cfg.min_valid_covalid:
            checked_pair.append(pair)
            checked_inlier.append(inlier)
    # collect the robsut mask inside the static
    fov_jump_pair_list = checked_pair
    inlier_list_jumped = torch.stack(checked_inlier, 0)
    # * assume fov is known, solve the optimal s,R,t between view pair and form an energy under such case
    # convert the [0,W], [0,H] to aspect un-distorted homo_list
    homo_list = track2undistroed_homo(track, height, width)

    e_list, fov_list = [], []
    search_candidates = torch.linspace(
        start=cfg.search_start, end=cfg.search_end, steps=cfg.search_N
    )[1:-1]
    for fov in tqdm(search_candidates):
        E, E_i, s_ij, R_ij, t_ij = compute_graph_energy_old(
            fov,
            fov_jump_pair_list,
            inlier_list_jumped.to(track.device),
            homo_list,
            dep_list,
            depth_decay_th=depth_decay_th,
            depth_decay_sigma=depth_decay_sigma,
        )
        e_list.append(E.item())
        fov_list.append(fov)
    e_list = np.array(e_list)
    best_ind = e_list.argmin()
    optimial_fov = fov_list[best_ind]
    # ! detect monotonic case and use fallback fov if no optimal is found
    if (e_list[1:] >= e_list[:-1]).all() or (e_list[1:] <= e_list[:-1]).all():
        logger.warning(
            f"FOV search mono case encountered, fall back to FOV={cfg.fallback_fov}"
        )
        optimial_fov = cfg.fallback_fov

    plt.figure(figsize=(10, 3))
    plt.plot(fov_list, e_list)
    plt.plot([optimial_fov, optimial_fov], [min(e_list), max(e_list)], "r--")
    plt.plot([optimial_fov], [min(e_list)], "o")
    plt.title(
        f"FOV Linear Search Best={optimial_fov:.3f} with energy {min(e_list):.6f}"
    )
    plt.xlabel("fov")
    plt.ylabel("ReprojEnergy")
    plt.tight_layout()
    viz_path = osp.join(ws, "viz", "moca")
    os.makedirs(viz_path, exist_ok=True)
    plt.savefig(osp.join(viz_path, "fov_search.jpg"))
    plt.close()
    logger.info(
        f"FOV search done, find FOV={optimial_fov:.3f} deg, figure saved at {osp.join(viz_path, 'fov_search.jpg')}"
    )

    return optimial_fov


def compute_graph_energy(
    fov,
    view_pair_list,
    pair_mask_list,
    homo_list: torch.Tensor,
    dep_list: torch.Tensor,
    depth_decay_th,
    depth_decay_sigma,
):
    # compute all the best aligned
    T, M = homo_list.shape[:2]
    rel_focal = 2 * fov_to_focal(torch.deg2rad(fov))
    cxcy_ratio = torch.tensor([0.5, 0.5])
    point_cam = backproject(
        homo_list.reshape(-1, 2), dep_list.reshape(-1), rel_focal, cxcy_ratio
    ).reshape(T, M, 3)

    ti_list = torch.as_tensor([p[0] for p in view_pair_list])
    tj_list = torch.as_tensor([p[1] for p in view_pair_list])

    xyz_cam_i, xyz_cam_j = point_cam[ti_list], point_cam[tj_list]
    # solve optimal
    # chunk operation

    # ! reweight the pair mask list
    max_dep = torch.max(xyz_cam_i[..., -1], xyz_cam_j[..., -1])
    robust_w = positive_th_gaussian_decay(max_dep, depth_decay_th, depth_decay_sigma)

    s_ji, R_ji, t_ji = compute_batch_optimal_sRt_ji(
        xyz_cam_i, xyz_cam_j, pair_mask_list.float() * robust_w.float()
    )

    assert (s_ji > 0).all(), "scale solution error, should be non-negative"
    # compute cross coordinates
    xyz_cam_j_from_i = (
        torch.einsum("bij,bnj->bni", R_ji, s_ji[:, None, None] * xyz_cam_i)
        + t_ji[:, None]
    )
    xyz_cam_i_from_j = torch.einsum(
        "bji,bnj->bni", R_ji, (xyz_cam_j - t_ji[:, None]) / s_ji[:, None, None]
    )
    # project and compute flow error
    uv_cam_j_from_i = project(xyz_cam_j_from_i, rel_focal, cxcy_ratio)  # B,N,2
    uv_cam_i_from_j = project(xyz_cam_i_from_j, rel_focal, cxcy_ratio)  # B,N,2
    rel_uv_track = homo_list[..., :2]
    gt_uv_i, gt_uv_j = rel_uv_track[ti_list], rel_uv_track[tj_list]
    uv_diff_at_i = (gt_uv_i - uv_cam_i_from_j).norm(dim=-1)
    uv_diff_at_j = (gt_uv_j - uv_cam_j_from_i).norm(dim=-1)
    E_i = (
        ((uv_diff_at_i + uv_diff_at_j) * pair_mask_list).sum(1)
        / pair_mask_list.sum(1)
        / 2.0
    )
    E = E_i.mean()
    # flip the order
    s_ij = 1 / s_ji
    R_ij = R_ji.permute(0, 2, 1)
    t_ij = -torch.einsum("nij,nj->ni", R_ij, t_ji) * s_ij[:, None]
    # return E, E_i, s_ji, R_ji, t_ji
    # order: xyz_i = sR xyz_j + t
    return E, E_i, s_ij, R_ij, t_ij


def track2undistroed_homo(track, H, W):
    # the short side is -1,1, the long side may exceed
    H, W = float(H), float(W)
    L = min(H, W)
    u, v = track[..., 0], track[..., 1]
    u = 2.0 * u / L - W / L
    v = 2.0 * v / L - H / L
    uv = torch.stack([u, v], -1)
    return uv


def compute_batch_optimal_sRt_ji(xyz_i, xyz_j, mask):
    # solve procrustes
    # j is q, i is p
    # q = sRp +t; xyz_j = sR xyz_i + t

    # xyz_i, xyz_j: B,N,3
    # mask: B,N

    assert xyz_i.ndim == 3 and xyz_i.shape[-1] == 3
    assert xyz_j.ndim == 3 and xyz_j.shape[-1] == 3
    assert xyz_i.shape == xyz_j.shape
    assert mask.ndim == 2
    mask = mask.float()
    # ! if want to reweight have to add to mask.
    if not mask.any(-1).all():
        logger.warning("No valid pair to compute, return ID")
        return (
            torch.ones(len(xyz_i)),
            torch.eye(3).repeat(len(xyz_i), 1, 1),
            torch.zeros(len(xyz_i), 3),
        )
    mask_normalized = mask.float() / mask.sum(dim=1, keepdim=True)

    p_bar = (xyz_i * mask_normalized[..., None]).sum(dim=1, keepdim=True)
    q_bar = (xyz_j * mask_normalized[..., None]).sum(dim=1, keepdim=True)
    p = xyz_i - p_bar
    q = xyz_j - q_bar

    W = torch.einsum("bni,bnj->bnij", p, q)
    W = (W * mask[..., None, None]).sum(1)  # B,3,3

    U, s, V = torch.svd(W.double())
    # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
    U, s, V = U.float(), s.float(), V.float()
    # R_star = V @ (U.T)
    # ! handling flipping
    R_tmp = torch.einsum("nij,nkj->nik", V, U)
    det = torch.det(R_tmp)
    dia = torch.ones(len(det), 3).to(det)
    dia[:, -1] = det
    Sigma = torch.diag_embed(dia)
    V = torch.einsum("nij,njk->nik", V, Sigma)
    R_star = torch.einsum("nij,nkj->nik", V, U)

    pp = ((p**2).sum(-1) * mask).sum(-1)
    s_star = s.sum(-1) / pp
    t_star = q_bar.squeeze(1) - torch.einsum(
        "bij,bj->bi", R_star, s_star[:, None] * (p_bar.squeeze(1))
    )
    # q = sRp +t; xyz_j = sR xyz_i + t
    return s_star, R_star, t_star


def make_pair_list(N, interval=[1], dense_flag=False, track_mask=None, min_valid_num=0):
    # N: int, number of frames
    # interval: list of int, interval between frames
    # dense_flag: bool, whether to use dense pair
    pair_list = []
    for T in interval:
        for i in range(N - T):
            if T > 1 and not dense_flag:
                if i % T != 0:
                    continue
            if track_mask is not None:
                # check the common visib
                valid_num = (track_mask[i] & track_mask[i + T]).sum()
                if valid_num < min_valid_num:
                    continue
            pair_list.append((i, i + T))
    return pair_list
