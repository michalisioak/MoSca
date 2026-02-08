import os
import os.path as osp
import numpy as np
from loguru import logger
import torch

from prior.tracking import TrackingConfig
from prior.tracking.identify import identify_tracks
from prior.tracking.viz import Visualizer


def save_tracks(
    img_list: torch.Tensor,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    ws: str,
    name: str,
    cfg: TrackingConfig,
):
    """
    img_list: torch.Tensor(T,H,W,3)

    tracks: torch.Tensor(T,N,2)

    visibility: torch.Tensor(T,N)
    """
    npz_folder = osp.join(ws, "tracks")
    _, height, width, _ = img_list.shape
    os.makedirs(npz_folder, exist_ok=True)
    s_mask, d_mask = identify_tracks(
        tracks, height=height, width=width, cfg=cfg.identify
    )
    np.savez_compressed(
        osp.join(ws, f"{npz_folder}/{name}.npz"),
        s_tracks=tracks[:, s_mask, :].cpu().numpy(),
        s_visibility=visibility[:, s_mask].cpu().numpy(),
        d_tracks=tracks[:, d_mask, :].cpu().numpy(),
        d_visibility=visibility[:, d_mask].cpu().numpy(),
    )
    logger.info(f"Saved tracks to {npz_folder}/{name}.npz")

    if not cfg.vizualize:
        return

    vis = Visualizer(save_dir=osp.join(ws, "viz", "tracks"))
    viz_choice = np.random.choice(
        tracks[:, s_mask, :].shape[0],
        min(tracks[:, s_mask, :].shape[0], cfg.max_viz_cnt),
    )
    vis.visualize(
        video=img_list.permute(0, 3, 1, 2)[None],
        tracks=tracks[:, s_mask, :][None, :, viz_choice, :2],
        visibility=visibility[:, s_mask][None, :, viz_choice],
        filename=f"static_{name}",
    )
    viz_choice = np.random.choice(
        tracks[:, d_mask, :].shape[0],
        min(tracks[:, d_mask, :].shape[0], cfg.max_viz_cnt),
    )
    vis.visualize(
        video=img_list.permute(0, 3, 1, 2)[None],
        tracks=tracks[:, d_mask, :][None, :, viz_choice, :2],
        visibility=visibility[:, d_mask][None, :, viz_choice],
        filename=f"dynamic_{name}",
    )
