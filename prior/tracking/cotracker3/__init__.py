from dataclasses import dataclass
from typing import cast
from loguru import logger
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from prior.tracking.viz import Visualizer

from prior.tracking import TrackingConfig
from prior.tracking.queries import get_uniform_random_queries
from prior.tracking.save import save_tracks
from prior.vid_loader import load_video

import torch.nn as nn


@dataclass
class CoTrackerConfig(TrackingConfig):
    online_flag: bool = True


@torch.no_grad()
def __online_inference_one_pass__(
    video_pt_cpu, queries_cpu, model, device, add_support_grid=True
):
    T = video_pt_cpu.shape[1]
    first_flag = True
    queries = queries_cpu.to(device)
    for i in tqdm(range(T)):
        if i % model.step == 0 and i > 0:
            video_chunk = video_pt_cpu[:, max(0, i - model.step * 2) : i].to(device)
            pred_tracks, pred_visibility = model(
                video_chunk,
                is_first_step=first_flag,
                queries=queries,
                add_support_grid=add_support_grid,
            )
            first_flag = False
    pred_tracks, pred_visibility = model(
        video_pt_cpu[:, -(T % model.step) - model.step - 1 :].to(device),
        False,
        queries=queries,
        add_support_grid=add_support_grid,
    )
    torch.cuda.empty_cache()
    return pred_tracks.cpu(), pred_visibility.cpu()


@torch.no_grad()
def online_track_point(video_pt, queries, model, device, add_support_grid=True):
    T = video_pt.shape[1]
    N = queries.shape[1]
    # * forward
    pred_tracks_fwd, pred_visibility_fwd = __online_inference_one_pass__(
        video_pt, queries, model, device, add_support_grid
    )
    pred_tracks_fwd = pred_tracks_fwd[0, :, :N]  # T,N,2
    pred_visibility_fwd = pred_visibility_fwd[0, :, :N]  # T,N
    # * inverse manually
    video_pt_inv = video_pt.flip(1)
    queries_inv = queries.clone()
    queries_inv[..., 0] = T - 1 - queries_inv[..., 0]
    pred_tracks_bwd, pred_visibility_bwd = __online_inference_one_pass__(
        video_pt_inv, queries_inv, model, device, add_support_grid
    )
    pred_tracks_bwd = pred_tracks_bwd.flip(1)[0, :, :N]  # T,N,2
    pred_visibility_bwd = pred_visibility_bwd.flip(1)[0, :, :N]  # T,N
    # * fuse the forward and backward
    bwd_mask = torch.arange(T)[:, None] < queries.cpu()[0, :, 0][None, :]  # T,N
    pred_tracks = torch.where(bwd_mask[..., None], pred_tracks_bwd, pred_tracks_fwd)
    pred_visibility = torch.where(bwd_mask, pred_visibility_bwd, pred_visibility_fwd)
    return pred_tracks, pred_visibility


@torch.inference_mode()
def cotracker3(cfg: CoTrackerConfig, ws: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("CoTracker3 proccessing...")
    if cfg.online_flag:
        postfix = "online"
    else:
        postfix = "offline"
    cotracker = cast(
        nn.Module,
        torch.hub.load("facebookresearch/co-tracker", f"cotracker3_{postfix}"),
    )
    cotracker.to(device)
    cotracker.eval()

    img_list = torch.from_numpy(load_video(ws))
    full_video_pt = img_list.permute(0, 3, 1, 2)[None].float()  # [B,T,3,H,W]
    _, full_T, _, H, W = full_video_pt.shape

    # sta cfg
    video_pt = full_video_pt.clone()
    T = full_T

    tracks = torch.zeros(T, 0, 2)
    visibility = torch.zeros(T, 0).bool()

    cur = 0
    tracks = []
    visibility = []
    while cur < cfg.total_points:
        logger.info(
            f"Processing {cur}-{cur + cfg.chunk_size}/{cfg.total_points} points..."
        )
        queries = get_uniform_random_queries(T, H, W, cfg.chunk_size)[None].to(device)
        if cfg.online_flag:
            _tracks, _visibility = online_track_point(
                video_pt, queries, cotracker, device
            )  # T,N,2; T,N
        else:
            _tracks, _visibility = cotracker(
                video_pt.to(device), queries.to(device), backward_tracking=True
            )
            _tracks, _visibility = _tracks[0].cpu(), _visibility[0].cpu()
        tracks.append(_tracks)
        visibility.append(_visibility)
        cur = cur + cfg.chunk_size
    tracks = torch.cat(tracks, dim=1)
    visibility = torch.cat(visibility, dim=1)

    viz_choice = np.random.choice(
        tracks.shape[1], min(tracks.shape[1], cfg.max_viz_cnt)
    )
    vis = Visualizer(save_dir=osp.join(ws, "viz", "tracks"))
    vis.visualize(
        video=video_pt,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"cotracker3_{postfix}",
    )

    save_tracks(
        tracks=tracks.permute(1, 0, 2).cpu().numpy(),  # N, T, 2
        visibility=visibility.permute(1, 0).cpu().numpy(),  # N, 1
        ws=ws,
        name=f"cotracker3_{postfix}",
    )
