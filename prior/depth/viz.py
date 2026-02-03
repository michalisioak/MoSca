import os
import os.path as osp
from loguru import logger
import numpy as np
from matplotlib import cm
import imageio


def viz_depth_list(depths: list, ws: str, name: str, viz_quantile=3):
    depth_folder = osp.join(ws, "viz", "depths")
    os.makedirs(depth_folder, exist_ok=True)
    dep_list = np.stack(depths, axis=0)
    dep_valid_mask = dep_list > 1e-6
    dep_max = np.percentile(dep_list[dep_valid_mask], 100 - viz_quantile)
    dep_min = np.percentile(dep_list[dep_valid_mask], viz_quantile)
    dep_list = np.clip(dep_list, dep_min, dep_max)
    dep_list = (dep_list - dep_min) / (dep_max - dep_min)
    dep_list[~dep_valid_mask] = 0
    viz_list = []
    for dep in dep_list:
        viz = cm.get_cmap("viridis")(dep)[:, :, :3]
        viz_list.append((viz * 255).astype(np.uint8))
    imageio.v3.imwrite(osp.join(depth_folder, f"{name}.mp4"), viz_list)
    logger.info(f"Saved depth visualization to {depth_folder}/{name}.mp4")
