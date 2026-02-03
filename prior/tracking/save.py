import os
import os.path as osp
import numpy as np
from loguru import logger


def save_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    ws: str,
    name: str,
):
    npz_folder = osp.join(ws, "tracks")
    os.makedirs(npz_folder, exist_ok=True)
    np.savez_compressed(
        osp.join(ws, f"{npz_folder}/{name}.npz"),
        tracks=tracks,
        visibility=visibility,
    )
    logger.info(f"Saved tracks to {npz_folder}/{name}.npz")
