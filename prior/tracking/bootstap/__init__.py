from dataclasses import dataclass
import os
import os.path as osp
import numpy as np
from tapnet.torch import tapir_model
from loguru import logger
import torch

from prior.tracking import TrackingConfig
from prior.tracking.bootstap.inference import inference
from prior.tracking.queries import get_uniform_random_queries
from prior.tracking.save import save_tracks

from prior.tracking.viz import Visualizer
from prior.vid_loader import load_video


@dataclass
class BootsTAPConfig(TrackingConfig):
    checkpoint: str = "bootstapir_checkpoint_v2"
    download_url: str = (
        "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
    )
    tap_size: int | None = None


@torch.inference_mode()
def bootstap(cfg: BootsTAPConfig, ws: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_folder = osp.join(os.getcwd(), "weights", "tapnet")
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = osp.join(checkpoint_folder, cfg.checkpoint + ".pt")
    if not osp.exists(checkpoint_path):
        logger.info(f"Downloading checkpoint from {cfg.download_url}...")
        wget_cmd = f"wget -P {checkpoint_folder} {cfg.download_url}"
        os.system(wget_cmd)
    model = tapir_model.TAPIR()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(device)

    logger.info("bootsTAP proccessing...")
    img_list = torch.from_numpy(load_video(ws)).to(device)
    T, H, W, C = img_list.shape

    queries = get_uniform_random_queries(T, H, W, cfg.total_points).to(device)
    queries = queries[:, [0, 2, 1]]

    cur = 0
    tracks, visibility = [], []
    while cur < cfg.total_points:
        logger.info(
            f"Processing {cur}-{cur + cfg.chunk_size}/{cfg.total_points} points..."
        )
        cur_queries = queries[cur : min(cur + cfg.chunk_size, len(queries))]
        _tracks, _visibility = inference(img_list, cur_queries, model)
        tracks.append(_tracks.cpu())
        visibility.append(_visibility.cpu())
        cur = cur + cfg.chunk_size
    tracks = torch.cat(tracks, dim=0)
    visibility = torch.cat(visibility, dim=0)

    save_tracks(
        tracks=tracks.cpu().numpy(),
        visibility=visibility.cpu().numpy(),
        ws=ws,
        name="bootsTAP",
    )
    viz_choice = np.random.choice(
        tracks.shape[0], min(tracks.shape[0], cfg.max_viz_cnt)
    )
    vis = Visualizer(save_dir=osp.join(ws, "viz", "tracks"))
    vis.visualize(
        video=img_list.permute(0, 3, 1, 2)[None],
        tracks=tracks.permute(1, 0, 2)[None, :, viz_choice, :2],
        visibility=visibility.permute(1, 0)[None, :, viz_choice],
        filename="bootstap",
    )

    return tracks, visibility
