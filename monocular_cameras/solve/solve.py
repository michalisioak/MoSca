from dataclasses import dataclass, field
import os.path as osp
from loguru import logger
import torch
import logging

from epi_helpers import analyze_track_epi
from .fov_search import find_initinsic, FovSearchConfig
from monocular_cameras.cameras import MonocularCameras
from bundle import query_buffers_by_track


from .bundle_adjastment import StaticBundleAdjastmentConfig, compute_static_ba


@dataclass
class MoCaConfig:
    robust_depth_decay_th = 2.0
    robust_depth_decay_sigma = 1.0
    robust_std_decay_th = 0.2
    robust_std_decay_sigma = 0.2
    iso_focal = False
    bundle_adjastment: StaticBundleAdjastmentConfig = field(
        default_factory=StaticBundleAdjastmentConfig
    )
    depth_filter_th: float | None = None
    depth_filter_min_cnt: int = 1
    init_cam_with_optimal_fov_results = True
    viz_valid_ba_points = False
    iso_focal: bool = False
    fov: FovSearchConfig = field(default_factory=FovSearchConfig)
    fov_min_valid_covalid = 64


def moca_solve(
    ws: str,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    s_tracks: torch.Tensor,
    cfg: MoCaConfig,
    device=torch.device("cuda:0"),
):
    torch.cuda.empty_cache()
    time, height, width, _ = rgb.shape

    depth_median = float(dep.median())
    logger.warning(
        f"All robust decay th and sigma are factors w.r.t the depth median, rescale them with median={depth_median:.2f}"
    )
    robust_depth_decay_th = cfg.robust_depth_decay_th * depth_median
    robust_depth_decay_sigma = cfg.robust_depth_decay_sigma * depth_median
    robust_std_decay_th = cfg.robust_std_decay_th * depth_median
    robust_std_decay_sigma = cfg.robust_std_decay_sigma * depth_median

    assert not torch.isinf(s_tracks).any()

    # * verify the static track maximum visible depth and filter out
    if cfg.depth_filter_th and cfg.depth_filter_th > 0:
        logging.warning(
            f"MoCa BA set to filter out very far tracks with th={cfg.depth_filter_th}"
        )
        s_track_dep = query_buffers_by_track(dep[..., None], s_tracks)
        invalid_mask = s_track_dep.squeeze(-1) > cfg.depth_filter_th
        # remove the tracks if there is not enough valid
        valid_cnt = (~invalid_mask).sum(dim=0)
        filter_mask = valid_cnt >= cfg.depth_filter_min_cnt
        # filter
        s_tracks = s_tracks[:, filter_mask]

    # * 2. compute also for later fov inlier mask
    # rerun the epi analysis for later FOV init, with the pairs that count the static common visible mask
    fov_jump_pair_list = make_pair_list(
        time,
        interval=cfg.fov_search_intervals,
        dense_flag=True,
        track_mask=sta_track_mask,
        min_valid_num=cfg.fov_min_valid_covalid,
    )
    assert len(fov_jump_pair_list) > 0, "no valid pair for FOV search"
    logging.info(f"Start analyzing {len(fov_jump_pair_list)} pairs for FOV search")
    _, _, inlier_list_jumped = analyze_track_epi(
        fov_jump_pair_list, s_tracks, sta_track_mask, H=height, W=width
    )
    checked_pair, checked_inlier = [], []
    for pair, inlier in zip(fov_jump_pair_list, inlier_list_jumped):
        if inlier.sum() > cfg.fov_min_valid_covalid:
            checked_pair.append(pair)
            checked_inlier.append(inlier)
    # collect the robsut mask inside the static
    fov_jump_pair_list = checked_pair
    inlier_list_jumped = torch.stack(checked_inlier, 0)

    # * 3. compute FOV
    optimal_fov, opt_R_ij, opt_t_ij = find_initinsic(
        ws=ws,
        height=height,
        width=width,
        pair_list=fov_jump_pair_list,
        pair_mask_list=inlier_list_jumped.to(
            device
        ),  # ! use the inlier mask to find the optimal fov
        track=s_tracks.to(device),
        dep_list=dep.to(device),
        cfg=cfg.fov,
        depth_decay_th=robust_depth_decay_th,
        depth_decay_sigma=robust_depth_decay_sigma,
    )
    # * 4. prepare camra
    # todo: ! warning, the scale is ignored during initalziation, let the BA to solve this
    cams = MonocularCameras(
        time,
        height,
        width,
        (optimal_fov, optimal_fov, 0.5, 0.5),
        delta_flag=True,
        init_camera_pose=(
            Rt2T(opt_R_ij, opt_t_ij) if cfg.init_cam_with_optimal_fov_results else None
        ),
        iso_focal=cfg.iso_focal,
    ).to(device)

    # * 5. bundle adjustment
    compute_static_ba(
        s2d=s2d,
        ws=ws,
        s_track=s_tracks.to(device),
        s_track_valid_mask=s_track_mask.to(device),
        cfg=cfg.bundle_adjastment,
        cams=cams,
        depth_decay_th=robust_depth_decay_th,
        std_decay_th=robust_std_decay_th,
        std_decay_sigma=robust_std_decay_sigma,
    )
    torch.cuda.empty_cache()
    return cams


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


def Rt2T(R_list, t_list):
    # R_list: N,3,3, t_list: N,3
    assert len(R_list) == len(t_list)
    assert R_list.ndim == 3 and t_list.ndim == 2
    N = len(R_list)
    ret = torch.cat([R_list, t_list[:, :, None]], dim=2)
    bottom = torch.Tensor([0, 0, 0, 1.0]).to(ret)
    ret = torch.cat([ret, bottom[None, None].expand(N, -1, -1)], dim=1)
    return ret
