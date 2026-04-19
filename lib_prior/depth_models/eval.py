import json
import logging

import numpy as np
import sys
import os
import os.path as osp

import torch
from tqdm import tqdm

import cv2


sys.path.append(osp.abspath(osp.dirname(__file__)))


def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error


def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5, open_ksize=3):
    # logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    dep_boundary_errors, dep_valid_masks = [], []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, error = detect_depth_occlusion_boundaries(dep, threshold, ksize)
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


def abs_rel(
    pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None
) -> float:
    """
    Compute Absolute Relative Difference (Abs Rel) between predicted and ground truth depth maps.

    Args:
        pred: Predicted depth map of shape (T, H, W) or (H, W)
        gt: Ground truth depth map of same shape as pred
        valid_mask: Optional boolean mask of same shape indicating valid pixels.
                    If None, all pixels are considered valid.

    Returns:
        Abs Rel value (lower is better)

    Formula:
        Abs Rel = mean(|pred - gt| / gt)
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    else:
        valid_mask = valid_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    # Filter invalid values
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return np.nan

    # Compute Abs Rel
    abs_rel_val = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

    return float(abs_rel_val)


def abs_rel_per_frame(
    pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute Abs Rel for each frame independently.

    Args:
        pred: Predicted depth map of shape (T, H, W)
        gt: Ground truth depth map of shape (T, H, W)
        valid_mask: Optional boolean mask of shape (T, H, W)

    Returns:
        Array of shape (T,) with Abs Rel per frame
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.ndim == 2:
        # Single image case
        return np.array([abs_rel(pred, gt, valid_mask)])

    T = pred.shape[0]
    results = np.zeros(T)

    for t in range(T):
        mask_t = valid_mask[t] if valid_mask is not None else None
        results[t] = abs_rel(pred[t], gt[t], mask_t)

    return results


def abs_rel_weighted(
    pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None
) -> float:
    """
    Compute weighted Abs Rel where each pixel contributes proportional to 1/gt.
    This is equivalent to mean(|pred - gt| / gt).

    Args:
        pred: Predicted depth map of shape (T, H, W)
        gt: Ground truth depth map of shape (T, H, W)
        valid_mask: Optional boolean mask

    Returns:
        Weighted Abs Rel value
    """
    pred = np.asarray(pred, dtype=np.float32).flatten()
    gt = np.asarray(gt, dtype=np.float32).flatten()

    if valid_mask is not None:
        valid_mask = valid_mask.flatten()
        pred = pred[valid_mask]
        gt = gt[valid_mask]
    else:
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

    if len(pred) == 0:
        return np.nan

    # Weighted by inverse of ground truth
    weights = 1.0 / gt
    abs_diff = np.abs(pred - gt)

    return float(np.sum(weights * abs_diff) / np.sum(weights))


def load_gt_dep(src: str):
    import os
    import os.path as osp

    img_fns = [
        f
        for f in os.listdir(osp.join(src, "images"))
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    img_fns.sort()
    img_names = [osp.splitext(f)[0] for f in img_fns]

    dep = [
        np.load(osp.join(src, "sensor_depth", f"{img_name}.npz"))["dep"]
        for img_name in img_names
    ]
    dep = np.stack(dep)  # T,H,W
    return dep


def load_iphone_cameras(dir_path):
    ######################################
    # ! dycheck use Nerfies convention, which is opencv format
    # https://github.com/google/nerfies#datasets
    # https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md
    ######################################

    all_fns = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    all_fns.sort()
    all_cam_ids = np.array([int(f.split("_")[0].split(".")[0]) for f in all_fns])
    all_cam_ids = np.unique(all_cam_ids).tolist()
    logging.info(f"Loading iphone cameras from {dir_path} with camid={all_cam_ids}")
    ret = {}
    for cam_id in all_cam_ids:
        cam_fns = [f for f in all_fns if f.startswith(f"{cam_id}")]
        cam_fns.sort()
        T_wi_list, fovdeg_list, t_list = [], [], []
        cxcy_ratio_list = []
        # ! iphone dataset only has one fov
        for fn in tqdm(cam_fns):
            time_ind = int(fn.split("_")[1].split(".")[0])  # ! time start from 0, not 1
            t_list.append(time_ind)

            with open(osp.join(dir_path, fn), "r") as f:
                cam_data = json.load(f)

            # ! for shape of motion camera
            if "image_size" not in cam_data:
                cam_data["image_size"] = [720, 960]  # W,H

            focal = cam_data["focal_length"]
            T_wi = np.eye(4)
            t_wi = np.asarray(cam_data["position"])
            R_wi = np.asarray(cam_data["orientation"]).T
            T_wi[:3, :3] = R_wi
            T_wi[:3, 3] = t_wi
            # world = R.T @ local
            # https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/camera.py#L263
            # the optical axis is the last row
            T_wi_list.append(torch.from_numpy(T_wi))

            # ! get the fov in current code format, the focal with the shortest side
            short_size = min(cam_data["image_size"])  # -1, +1
            fovdeg = np.rad2deg(2 * np.arctan(short_size / (2 * focal)))

            # also load camera center
            principal_point = np.asarray(cam_data["principal_point"])
            image_size = np.asarray(cam_data["image_size"])
            cx_ratio = principal_point[0] / image_size[0]
            cy_ratio = principal_point[1] / image_size[1]
            cxcy_ratio_list.append([cx_ratio, cy_ratio])

            fovdeg_list.append(fovdeg)
        ret[cam_id] = {
            "T_wi": torch.stack(T_wi_list, 0).float(),
            "fovdeg": fovdeg_list,
            "cxcy_ratio": cxcy_ratio_list,
            "t": t_list,
            "filenames": [f[:-5] for f in cam_fns],
        }
    return ret


def load_iphone_gt_poses(src, t_subsample):
    gt_training_cams = load_iphone_cameras(osp.join(src, "cameras"))
    assert len(gt_training_cams) == 1, (
        "Only support Mono camera for now in Iphone dataset"
    )
    # subsample the gt camera
    gt_training_cams = {k: v[::t_subsample] for k, v in gt_training_cams[0].items()}
    gt_testing_cams = load_iphone_cameras(osp.join(src, "test_cameras"))
    gt_training_fov = gt_training_cams["fovdeg"][0]
    gt_training_cam_T_wi = gt_training_cams["T_wi"]
    # ! this bound which test frames can be retrieved
    gt_training_tids = gt_training_cams["t"]
    gt_training_cxcy_ratio = gt_training_cams["cxcy_ratio"]

    gt_testing_cam_T_wi_list, gt_testing_tids_list = [], []
    gt_testing_fov_list, gt_testing_fns_list = [], []
    gt_testing_cxcy_ratio_list = []
    for it in gt_testing_cams.values():
        sample_index = [i for i, t in enumerate(it["t"]) if t in gt_training_tids]
        gt_testing_tids = [gt_training_tids.index(it["t"][i]) for i in sample_index]
        gt_testing_tids_list.append(gt_testing_tids)
        gt_testing_cam_T_wi = it["T_wi"][sample_index]
        gt_testing_cam_T_wi_list.append(gt_testing_cam_T_wi)
        gt_testing_fns_list.append([it["filenames"][i] for i in sample_index])
        # ! assume all cam stays the same across time, use the first one
        gt_testing_fov_list.append(it["fovdeg"][0])
        gt_testing_cxcy_ratio_list.append(it["cxcy_ratio"][0])
    # todo: can use gt camera for evaluation
    return (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    )


if __name__ == "__main__":
    device = "cuda"
    src = "/home/MoSca/data/iphone/spin"

    import imageio
    from eval import abs_rel, load_gt_dep, load_iphone_gt_poses

    img_src = osp.expanduser(osp.join(src, "images"))
    fns = os.listdir(img_src)
    fns.sort()
    fn_list = [fn for fn in fns if fn.endswith(".jpg") or fn.endswith(".png")]
    img_list = np.array([imageio.imread(img_src + "/" + fn) for fn in fn_list])
    (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    ) = load_iphone_gt_poses(src, 1)
    gt_fovdeg = float(gt_training_fov)
    cxcy_ratio = gt_training_cxcy_ratio[0]  # gt camera center
    H = img_list[0].shape[0]
    W = img_list[0].shape[1]

    focal_length = (H / 2) / np.tan(np.radians(gt_fovdeg / 2))
    K = np.asarray(
        [
            [focal_length, 0, H * cxcy_ratio[0]],
            [0, focal_length, W * cxcy_ratio[1]],
            [0, 0, 1],
        ]
    )
    extrs = gt_training_cam_T_wi

    deps = [
        np.load(
            osp.join(src, "depth_anything_depth", f"{osp.splitext(img_name)[0]}.npz")
        )["dep"]
        for img_name in fn_list
    ]
    deps = np.stack(deps)  # T,H,W
    gt_deps = load_gt_dep(src)

    print("Abs Rel:", abs_rel(np.asarray(deps), np.asarray(gt_deps)))
