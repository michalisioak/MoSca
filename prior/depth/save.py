from loguru import logger
import numpy as np
import os
import os.path as osp


def save_depth_list(dep_list: list, ws: str, name: str, invalid_mask_list=None):
    npz_folder = osp.join(ws, "depths")
    os.makedirs(npz_folder, exist_ok=True)
    for i in range(len(dep_list)):
        if invalid_mask_list is not None:
            dep_list[i][invalid_mask_list[i] > 0] = 0
    np.savez_compressed(f"{npz_folder}/{name}.npz", dep=dep_list)
    logger.info(f"Saved depth maps to {npz_folder}/{name}.npz")


def save_camera(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    ws: str,
    name: str,
):
    npz_folder = osp.join(ws, "camera")
    os.makedirs(npz_folder, exist_ok=True)
    np.savez_compressed(
        f"{npz_folder}/{name}.npz", extrinsics=extrinsics, intrinsics=intrinsics
    )
    logger.info(f"Saved camera to {npz_folder}/{name}.npz")
