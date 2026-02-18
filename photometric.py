from typing import Literal
import numpy as np
import torch
import os.path as osp

from lib_mosca.mosca import MoSca


# from monocular_cameras.cameras import MonocularCameras
from monocular_cameras.cameras import MonocularCameras
from prior.vid_loader import load_video
from scene.cfg import GaussianSplattingConfig
from scene.fetch import LeavesFetchConfig
from scene.fg_mask import identify_fg_mask_by_nearest_curve
from scene.fit import photometric_fit, PhotometricFitConfig
from scene.static import create_static_model


def photometric_reconstruct(
    ws: str,
    gs_cfg: GaussianSplattingConfig,
    fetch_cfg: LeavesFetchConfig,
    photo_cfg: PhotometricFitConfig,
    tracks: Literal["cotracker3_offline", "cotracker3_online"] = "cotracker3_offline",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rgb = torch.from_numpy(load_video(ws)).to(device)
    dep = torch.from_numpy(np.load(osp.join(ws, "depths", "bundle.npz"))["dep"]).to(
        device
    )

    tracks_npz = np.load(osp.join(ws, "tracks", f"{tracks}.npz"))

    s_tracks = torch.from_numpy(tracks_npz["s_tracks"]).to(device)
    s_visibility = torch.from_numpy(tracks_npz["s_visibility"]).to(device)
    d_tracks = torch.from_numpy(tracks_npz["d_tracks"]).to(device)
    d_visibility = torch.from_numpy(tracks_npz["d_visibility"]).to(device)

    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(ws, "bundle", "cams.pth"))
    ).to(device)
    scaffold = MoSca.load_from_ckpt(torch.load(osp.join(ws, "mosca", "mosca.pth"))).to(
        device
    )

    # * reset the scaffold mlevel config
    # scaffold.set_multi_level(
    #     mlevel_arap_flag=True,
    #     mlevel_list=getattr(fit_cfg, "photo_mlevel_list", [1, 6]),
    #     mlevel_k_list=getattr(fit_cfg, "photo_mlevel_k_list", [16, 8]),
    #     mlevel_w_list=getattr(fit_cfg, "photo_mlevel_w_list", [0.4, 0.3]),
    # )

    fg_mask = identify_fg_mask_by_nearest_curve(
        ws=ws,
        rgb=rgb,
        dep=dep,
        cams=cams,
        s_track=s_tracks,
        s_track_mask=s_visibility,
        d_track=d_tracks,
        d_track_mask=d_visibility,
    )

    s_model = create_static_model(
        dep=dep,
        rgb=rgb,
        gs_cfg=gs_cfg,
        cams=cams,
        fetch_cfg=fetch_cfg,
        gather_mask=~fg_mask,
    )

    d_model = create_dynamic_model(
        s2d=s2d,
        cams=cams,
        scf=scaffold,
        n_init=getattr(fit_cfg, "gs_dynamic_n_init", 30000),
        radius_max=getattr(fit_cfg, "gs_radius_max", 0.1),
        max_sph_order=getattr(fit_cfg, "gs_max_sph_order", 0),
        leaf_local_flag=getattr(fit_cfg, "gs_leaf_local_flag", True),
        nn_fusion=getattr(fit_cfg, "gs_nn_fusion", -1),
        # ! below is set to dyn_gs_model becaues it controls the densification
        max_node_num=getattr(fit_cfg, "gs_max_node_num", 100000),
    )

    photometric_fit(
        ws=ws,
        dep=dep,
        rgb=rgb,
        cams=cams,
        s_model=s_model,
        d_model=d_model,
        cfg=photo_cfg,
    )


if __name__ == "__main__":
    import tyro
    from seed import seed_everything

    seed_everything()
    tyro.cli(photometric_reconstruct)
