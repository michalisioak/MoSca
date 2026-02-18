from dataclasses import dataclass, field
import os
from typing import Literal
import imageio
from loguru import logger
import numpy as np
import torch

from monocular_cameras.cameras import MonocularCameras
from motion_scaffold import MoSca, MoScaConfig
import os.path as osp

from motion_scaffold.dynamic_curves import lift_3d
from motion_scaffold.fit import ScaffoldFitConfig, scaffold_fit
from prior.vid_loader import load_video
from recon_utils import viz_mosca_curves_before_optim
from viz_utils import viz_list_of_colored_points_in_cam_frame


@dataclass
class ScaffoldReconstructConfig:
    geo_keyframe_rate: int = 4
    geo_mosca_steps: int = 2_000
    refilter_curve_by_photo_error_cnt: int | None = None
    consider_photo_error_dyn_id_th: int | None = None
    mosca: MoScaConfig = field(default_factory=MoScaConfig)
    refilter_curve_by_photo_error_th = 0.1
    geo_mosca_use_mask_topo = True
    mosca_resample_flag = True
    consider_photo_error_dyn_id_open_ksize = -1
    get_dynamic_curves_filter_factor_in_world = True
    fit: ScaffoldFitConfig = field(default_factory=ScaffoldFitConfig)


def scaffold_reconstruct(
    ws: str,
    cfg: ScaffoldReconstructConfig,
    tracks: Literal["cotracker3_offline", "cotracker3_online"] = "cotracker3_offline",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mosca_viz = osp.join(ws, "viz", "mosca")
    os.makedirs(mosca_viz, exist_ok=True)

    rgb = torch.from_numpy(load_video(ws)).to(device)
    time, height, width, channels = rgb.shape
    dep = torch.from_numpy(np.load(osp.join(ws, "depths", "bundle.npz"))["dep"]).to(
        device
    )

    tracks_npz = np.load(osp.join(ws, "tracks", f"{tracks}.npz"))

    d_tracks = torch.from_numpy(tracks_npz["d_tracks"]).to(device)
    d_visibility = torch.from_numpy(tracks_npz["d_visibility"]).to(device)

    # if s2d.has_epi:
    #     viz_epi_mask = s2d.epi > EPI_TH
    #     viz_epi_mask = viz_epi_mask[..., None] * s2d.rgb
    #     imageio.mimsave(
    #         osp.join(log_path, f"epi_th={EPI_TH}_hardmask.gif"),
    #         (viz_epi_mask.cpu().numpy() * 255).astype(np.uint8),
    #     )

    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(ws, "bundle", "cams.pth"))
    ).to(device)

    curve_xyz, curve_mask, curve_rgb = lift_3d(
        cams=cams, rgb=rgb, dep=dep, track=d_tracks, track_mask=d_visibility
    )
    curve_rgb = (curve_rgb * curve_mask.unsqueeze(-1)).sum(0) / (
        curve_mask.sum(0).unsqueeze(-1) + 1e-3
    )

    viz_mosca_curves_before_optim(curve_xyz, curve_rgb, curve_mask, cams, mosca_viz)

    # * get scaffold
    scaffold: MoSca = MoSca(
        node_xyz=curve_xyz.detach().clone(),
        node_certain=curve_mask,
        cfg=cfg.mosca,
    )
    scaffold.compute_rotation_from_xyz()
    if cfg.mosca_resample_flag:
        sampled_inds = scaffold.resample_node(resample_factor=1.0, use_mask=True)
    else:
        logger.warning("Not resampling the scaffold")
        sampled_inds = torch.arange(scaffold.M).to(device)
    node_rgb = curve_rgb[sampled_inds]

    logger.info(
        f"MoSca: get scaffold with M={scaffold.M} and unit={scaffold.spatial_unit}"
    )

    logger.info("*" * 20 + "MoSca Geo" + "*" * 20)
    # * Optimize the curve with ARAP
    assert cfg.geo_mosca_use_mask_topo or d_tracks.shape[-1] == 3, (
        "Must use mask topo for 2D tracks"
    )
    if cfg.geo_mosca_steps > 0:
        scaffold = scaffold_fit(ws=ws, scf=scaffold, cfg=cfg.fit)
    viz_list = viz_list_of_colored_points_in_cam_frame(
        [cams.trans_pts_to_cam(t, it).cpu() for t, it in enumerate(scaffold.node_xyz)],
        node_rgb,
        zoom_out_factor=1.0,
    )
    imageio.v3.imwrite(
        osp.join(mosca_viz, "cam_curve_optimized.gif"), viz_list, loop=1000
    )

    os.makedirs(osp.join(ws, "mosca"), exist_ok=True)
    torch.save(scaffold.state_dict(), osp.join(ws, "mosca", "mosca.pth"))


if __name__ == "__main__":
    import tyro
    from seed import seed_everything

    seed_everything()
    tyro.cli(scaffold_reconstruct)
