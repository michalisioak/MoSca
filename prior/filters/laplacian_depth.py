import cv2
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class LaplacianDepthFilterConfig:
    threshold_ratio: float = 0.5
    ksize:int = 5
    open_ksize: int = 3

@torch.no_grad()
def laplacian_filter_depth(depths:np.ndarray, cfg: LaplacianDepthFilterConfig):
    # logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * cfg.threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, cfg.ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        # ! do a morph operator to remove outliers
        mask_opened = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel
        )
        mask_opened = mask_opened > 0
        # mask_opened = mask
        dep_valid_masks.append(mask_opened)
        dep_boundary_errors.append(error)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_boundary_errors = np.stack(dep_boundary_errors, axis=0)
    return dep_valid_masks, dep_boundary_errors

def detect_depth_occlusion_boundaries(depth_map:np.ndarray, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error