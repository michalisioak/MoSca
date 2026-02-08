from dataclasses import dataclass
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
from loguru import logger
import torch

from prior.depth.save import save_camera, save_depth_list
from prior.depth.viz import viz_depth_list
from prior.vid_loader import load_video


@dataclass
class DepthAnythingV3Config:
    model: str = "depth-anything/DA3METRIC-LARGE"


@torch.inference_mode()
def depth_anything_v3(cfg: DepthAnythingV3Config, ws: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthAnything3.from_pretrained(cfg.model)
    model.eval()
    model.to(device)
    logger.info("Depth-Anything processing...")
    img_list = load_video(ws)
    T, H, W, C = img_list.shape
    res = model.inference(
        list(img_list),
        ref_view_strategy="middle",
        process_res=max(H, W),
    )
    dep_list_unscaled = res.depth
    dep_list = []
    for dep in dep_list_unscaled:
        dep = cv2.resize(dep, (H, W), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    save_depth_list(dep_list, ws, "DepthAnything3")
    viz_depth_list(dep_list, ws, "DepthAnything3")
    if res.extrinsics is not None and res.intrinsics is not None:
        save_camera(res.extrinsics, res.intrinsics, ws, "DepthAnything3")
    return np.asarray(dep_list), res.extrinsics, res.intrinsics
