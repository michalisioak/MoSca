from dataclasses import dataclass, field
from loguru import logger
import torch
import logging

from monocular_cameras.solve.buffer_utils import query_buffers_by_track

from .fov_search import (
    compute_graph_energy,
    find_initinsic,
    FovSearchConfig,
    track2undistroed_homo,
)
from monocular_cameras.cameras import MonocularCameras


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


def moca_solve(
    ws: str,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    s_tracks: torch.Tensor,
    s_visibility: torch.Tensor,
    cfg: MoCaConfig,
    device: str,
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

    s_track_dep = query_buffers_by_track(dep[..., None], s_tracks, s_visibility)

    assert not torch.isinf(s_tracks).any()
    assert not torch.isnan(s_visibility).any()

    # * verify the static track maximum visible depth and filter out
    if cfg.depth_filter_th and cfg.depth_filter_th > 0:
        logging.warning(
            f"MoCa BA set to filter out very far tracks with th={cfg.depth_filter_th}"
        )
        invalid_mask = s_track_dep.squeeze(-1) > cfg.depth_filter_th
        # remove the tracks if there is not enough valid
        valid_cnt = (~invalid_mask).sum(dim=0)
        filter_mask = valid_cnt >= cfg.depth_filter_min_cnt
        # filter
        s_tracks = s_tracks[:, filter_mask]
        s_visibility = s_visibility[:, filter_mask]
        s_track_dep = s_track_dep[:, filter_mask]

    # * 2. compute also for later fov inlier mask
    # rerun the epi analysis for later FOV init, with the pairs that count the static common visible mask

    # * 3. compute FOV
    optimal_fov = find_initinsic(
        ws=ws,
        time=time,
        height=height,
        width=width,
        track_mask=s_visibility,
        track=s_tracks.to(device),
        dep_list=s_track_dep.to(device),
        cfg=cfg.fov,
        depth_decay_th=robust_depth_decay_th,
        depth_decay_sigma=robust_depth_decay_sigma,
    )
    # * 4. prepare camra
    # todo: ! warning, the scale is ignored during initalziation, let the BA to solve this
    continuous_pair_list = [(i, i + 1) for i in range(time - 1)]
    _, _, _, opt_R_ij, opt_t_ij = compute_graph_energy(
        optimal_fov,
        continuous_pair_list,
        s_visibility[[it[0] for it in continuous_pair_list]]
        * s_visibility[[it[1] for it in continuous_pair_list]],
        track2undistroed_homo(s_tracks, height, width),
        s_track_dep,
        depth_decay_th=robust_depth_decay_th,
        depth_decay_sigma=robust_depth_decay_sigma,
    )
    cams = MonocularCameras(
        time,
        height,
        width,
        (optimal_fov, optimal_fov, 0.5, 0.5),
        delta_flag=True,
        init_camera_pose=(
            Rt2T(opt_R_ij, opt_t_ij).cpu()
            if cfg.init_cam_with_optimal_fov_results
            else None
        ),
        iso_focal=cfg.iso_focal,
    ).to(device)

    # * 5. bundle adjustment
    compute_static_ba(
        rgb=rgb,
        dep=dep,
        ws=ws,
        s_track=s_tracks.to(device),
        track_mask=s_visibility.to(device),
        cfg=cfg.bundle_adjastment,
        cams=cams,
        depth_decay_th=robust_depth_decay_th,
        std_decay_th=robust_std_decay_th,
        std_decay_sigma=robust_std_decay_sigma,
    )
    torch.cuda.empty_cache()
    return cams


def Rt2T(R_list, t_list):
    # R_list: N,3,3, t_list: N,3
    assert len(R_list) == len(t_list)
    assert R_list.ndim == 3 and t_list.ndim == 2
    N = len(R_list)
    ret = torch.cat([R_list, t_list[:, :, None]], dim=2)
    bottom = torch.Tensor([0, 0, 0, 1.0]).to(ret)
    ret = torch.cat([ret, bottom[None, None].expand(N, -1, -1)], dim=1)
    return ret
