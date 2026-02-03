from dataclasses import dataclass, field
import os.path as osp
import torch
import numpy as np
import logging

from epi_helpers import analyze_track_epi, identify_tracks
from .fov_search import find_initinsic, FovSearchConfig
from intrinsic_helpers import compute_graph_energy, track2undistroed_homo
from monocular_cameras.cameras import MonocularCameras
from bundle import (
    query_buffers_by_track,
    prepare_track_homo_dep_rgb_buffers,
)
from moca_misc import make_pair_list, Rt2T, get_all_world_pts_list

from prior.loading import Saved2D

from .bundle_adjastment import StaticBundleAdjastmentConfig, compute_static_ba


@dataclass
class MoCaConfig:
    epi_th = EPI_TH
    ba_total_steps: int = 2000
    ba_switch_to_ind_step = 500
    ba_depth_correction_after_step = 500
    ba_max_frames_per_step = (32,)
    static_id_mode = ("raft" if s2d.has_epi else "track",)
    robust_depth_decay_th = 2.0
    robust_depth_decay_sigma = 1.0
    robust_std_decay_th = 0.2
    robust_std_decay_sigma = 0.2
    gt_cam: bool = False
    iso_focal = False
    rescale_gt_cam_transl = False
    bundle_adjastment: StaticBundleAdjastmentConfig = field(
        default_factory=StaticBundleAdjastmentConfig
    )
    depth_filter_th: float | None = None
    init_cam_with_optimal_fov_results = True
    viz_valid_ba_points = False
    iso_focal: bool = False
    fov: FovSearchConfig = field(default_factory=FovSearchConfig)


def moca_solve(
    ws,
    s2d: Saved2D,
    cfg: MoCaConfig,
    device=torch.device("cuda:0"),
    # EPI
    epi_th=1e-4,
):
    # configure_logging(osp.join(ws, "moca_solve.log"), debug=False)
    torch.cuda.empty_cache()
    s2d.to(device)
    H, W, T = s2d.H, s2d.W, s2d.T

    track = s2d.track.cpu()
    track_mask = s2d.track_mask.cpu()
    continuous_pair_list = make_pair_list(T, interval=[1], dense_flag=True)

    depth_median = float(s2d.dep[s2d.dep_mask].median())
    logging.warning(
        f"All robust decay th and sigma are factors w.r.t the depth median, rescale them with median={depth_median:.2f}"
    )
    robust_depth_decay_th = robust_depth_decay_th * depth_median
    robust_depth_decay_sigma = robust_depth_decay_sigma * depth_median
    robust_std_decay_th = robust_std_decay_th * depth_median
    robust_std_decay_sigma = robust_std_decay_sigma * depth_median

    # * 1. mark static track
    epierr_list = None
    if static_id_mode == "raft":
        logging.info("Use pre-computed 2D epi error to mark static tracks")
        # * mark by collecting epi error
        # collect the epi error from the pre-compute 2D epi from raft
        raft_epi = s2d.epi.clone()
        with torch.no_grad():
            epierr_list = query_buffers_by_track(raft_epi[..., None], track, track_mask)
            epierr_list = epierr_list.squeeze(-1).cpu()
    elif static_id_mode == "track":
        logging.info("Analyze the track epi to mark static tracks")
        # * mark by compute all epi for neighboring pairs and
        # first call for neighbor pairs, solve the epi, and get F, so later can use to initialize pose
        F_list, epierr_list, _ = analyze_track_epi(
            continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        )
        np.savez(
            osp.join(ws, "tracker_epi.npz"),
            continuous_pair_list=continuous_pair_list,
            F_list=F_list,
        )
    assert epierr_list is not None
    track_static_selection, track_dynamic_selection = identify_tracks(
        epierr_list, epi_th
    )

    s2d.register_track_indentification(track_static_selection, track_dynamic_selection)
    sta_track = track[:, track_static_selection]
    sta_track_mask = track_mask[:, track_static_selection]

    # if sta_track.shape[-1] == 3:
    #     logging.info(f"SpaT mode, direct use 3D Track")
    #     sta_track_dep = sta_track[..., -1].clone()
    # else:
    # ! looks like the spatracker depth is not reliable
    sta_track_dep = query_buffers_by_track(
        s2d.dep[..., None], sta_track, sta_track_mask
    ).to(sta_track_mask.device)

    assert not torch.isinf(sta_track_dep[sta_track_mask]).any()
    assert not torch.isnan(sta_track_dep[sta_track_mask]).any()

    # * verify the static track maximum visible depth and filter out
    if depth_filter_th > 0:
        logging.warning(
            f"MoCa BA set to filter out very far tracks with th={depth_filter_th}"
        )
        sta_track_dep = query_buffers_by_track(
            s2d.dep[..., None], sta_track, sta_track_mask
        ).to(sta_track_mask.device)
        invalid_mask = sta_track_dep.squeeze(-1) > depth_filter_th
        sta_track_mask = sta_track_mask * ~invalid_mask
        # remove the tracks if there is not enough valid
        valid_cnt = sta_track_mask.sum(dim=0)
        filter_mask = valid_cnt >= depth_filter_min_cnt
        # filter
        sta_track = sta_track[:, filter_mask]
        sta_track_mask = sta_track_mask[:, filter_mask]
        sta_track_dep = sta_track_dep[:, filter_mask]

    # * 2. compute also for later fov inlier mask
    # rerun the epi analysis for later FOV init, with the pairs that count the static common visible mask
    fov_jump_pair_list = make_pair_list(
        T,
        interval=fov_search_intervals,
        dense_flag=True,
        track_mask=sta_track_mask,
        min_valid_num=fov_min_valid_covalid,
    )
    assert len(fov_jump_pair_list) > 0, "no valid pair for FOV search"
    logging.info(f"Start analyzing {len(fov_jump_pair_list)} pairs for FOV search")
    _, _, inlier_list_jumped = analyze_track_epi(
        fov_jump_pair_list, sta_track, sta_track_mask, H=s2d.H, W=s2d.W
    )
    checked_pair, checked_inlier = [], []
    for pair, inlier in zip(fov_jump_pair_list, inlier_list_jumped):
        if inlier.sum() > fov_min_valid_covalid:
            checked_pair.append(pair)
            checked_inlier.append(inlier)
    # collect the robsut mask inside the static
    fov_jump_pair_list = checked_pair
    inlier_list_jumped = torch.stack(checked_inlier, 0)

    # * 3. compute FOV
    optimal_fov = find_initinsic(
        H=s2d.H,
        W=s2d.W,
        pair_list=fov_jump_pair_list,
        pair_mask_list=inlier_list_jumped.to(
            device
        ),  # ! use the inlier mask to find the optimal fov
        track=sta_track.to(device),
        dep_list=sta_track_dep.to(device),
        viz_fn=osp.join(ws, "fov_search.jpg"),
        cfg=cfg.fov,
        depth_decay_th=robust_depth_decay_th,
        depth_decay_sigma=robust_depth_decay_th,
    )
    # solve the neighboring pair list as well
    E, E_i, opt_s_ij, opt_R_ij, opt_t_ij = compute_graph_energy(
        optimal_fov,
        continuous_pair_list,
        sta_track_mask[[it[0] for it in continuous_pair_list]]
        * sta_track_mask[[it[1] for it in continuous_pair_list]],
        track2undistroed_homo(sta_track, H, W),
        sta_track_dep,
        depth_decay_th=robust_depth_decay_th,
        depth_decay_sigma=robust_depth_decay_sigma,
    )  # ! this is in metric space

    # * 4. prepare camra
    # todo: ! warning, the scale is ignored during initalziation, let the BA to solve this
    cams = MonocularCameras(
        s2d.T,
        s2d.H,
        s2d.W,
        [optimal_fov, optimal_fov, 0.5, 0.5],
        delta_flag=True,
        init_camera_pose=(
            Rt2T(opt_R_ij, opt_t_ij) if init_cam_with_optimal_fov_results else None
        ),
        iso_focal=cfg.iso_focal,
    ).to(device)

    # * 5. bundle adjustment
    compute_static_ba(
        s2d=s2d,
        log_dir=osp.join(ws, "bundle"),
        s_track=sta_track.to(device),
        s_track_valid_mask=sta_track_mask.to(device),
        cfg=cfg.bundle_adjastment,
        cams=cams,
        depth_decay_th=robust_depth_decay_th,
        std_decay_th=robust_std_decay_th,
        std_decay_sigma=robust_std_decay_sigma,
    )
    torch.cuda.empty_cache()

    # * 6 finish
    viz_all_pts = get_all_world_pts_list(
        s2d.homo_map, s2d.dep, s2d.rgb, s2d.dep_mask, cams
    )
    viz_all_pts = torch.cat(viz_all_pts, 0).numpy()
    viz_all_sel = np.random.choice(len(viz_all_pts), 50_000, replace=False)
    viz_all_pts = viz_all_pts[viz_all_sel]
    viz_all_pts[:, 3:] = viz_all_pts[:, 3:] * 255
    np.savetxt(osp.join(ws, "bundle", "all_pts.xyz"), viz_all_pts[:, :6], fmt="%.5f")

    return cams, s2d, track_static_selection
