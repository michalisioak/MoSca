from typing import Literal
import numpy as np
import torch
from loguru import logger
import os.path as osp

from monocular_cameras.solve import moca_solve, MoCaConfig
from prior.vid_loader import load_video


def static_reconstruct(
    ws: str,
    cfg: MoCaConfig,
    depth: Literal["DepthAnything3"] = "DepthAnything3",
    tracks: Literal["cotracker3_offline", "cotracker3_online"] = "cotracker3_offline",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("*" * 20 + "MoCa BA" + "*" * 20)
    rgb = torch.from_numpy(load_video(ws)).to(device)
    dep = torch.from_numpy(np.load(osp.join(ws, "depths", f"{depth}.npz"))["dep"]).to(
        device
    )

    tracks_npz = np.load(osp.join(ws, "tracks", f"{tracks}.npz"))

    s_tracks = torch.from_numpy(tracks_npz["s_tracks"]).to(device)
    s_visibility = torch.from_numpy(tracks_npz["s_visibility"]).to(device)

    return moca_solve(
        ws=ws,
        rgb=rgb,
        dep=dep,
        s_tracks=s_tracks,
        s_visibility=s_visibility,
        device=device,
        cfg=cfg,
    )


if __name__ == "__main__":
    import tyro
    from seed import seed_everything

    seed_everything()
    tyro.cli(static_reconstruct)
