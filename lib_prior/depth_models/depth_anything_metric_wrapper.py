import os
import os.path as osp
import sys
import cv2
from numpy import ndarray
import numpy as np


sys.path.append(osp.abspath(osp.dirname(__file__)))

import os.path as osp

import torch

from depth_anything_3.api import DepthAnything3
from depth_utils import viz_depth_list, save_depth_list


def get_depth_anything_model(device="cuda"):
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model.to(device=device)
    model.eval()
    return model


@torch.no_grad()
def depth_anything_proccess_folder(
    model: DepthAnything3,
    img_list: ndarray,
    fn_list: list[str],
    dst: str,
    invalid_mask_list=None,
    intrs: ndarray | None = None,
    extrs: ndarray | None = None,
):
    print("Depth-Anything processing...")
    os.makedirs(dst, exist_ok=True)
    print(f"Processing {img_list.shape} images...")
    T, H, W, C = img_list.shape
    assert intrs is not None, "Intrinsics are required for Depth-Anything Metric model"
    assert extrs is not None, "Extrinsics are required for Depth-Anything Metric model"
    pred = model.inference(
        list(img_list),
        ref_view_strategy="middle",
        process_res=max(H, W),
        # intrinsics=np.array([intrs] * len(img_list)) if intrs is not None else None,
        # extrinsics=extrs,
        use_ray_pose=True,
    )
    assert intrs is not None, "Intrinsics are required for Depth-Anything Metric model"
    focal_pixels = (intrs[0, 0] + intrs[1, 1]) / 2.0

    local_dep_list = pred.depth
    for i in range(len(local_dep_list)):
        local_dep_list[i] = focal_pixels * local_dep_list[i] / 300.0
    dep_list = []
    for i in range(len(pred.depth)):
        dep_list.append(
            cv2.resize(pred.depth[i], (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        )
    print(f"Saving depth maps... for {len(img_list)} frames")
    save_depth_list(dep_list, fn_list, dst, invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    torch.cuda.empty_cache()
    return dep_list


if __name__ == "__main__":
    device = "cuda"
    src = "/home/MoSca/data/iphone/spin"

    import imageio
    from eval import abs_rel, load_gt_dep, load_iphone_gt_poses, laplacian_filter_depth
    from viz import save_error_video_colormap

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
    extrs = gt_training_cam_T_wi.cpu().numpy()

    model = get_depth_anything_model(device=device)
    deps = depth_anything_proccess_folder(
        model,
        fn_list=fn_list,
        img_list=img_list,
        dst=osp.join(src, "../debug/depth_anything"),
        extrs=extrs,
        intrs=K,
    )
    gt_deps = load_gt_dep(src)

    print("Abs Rel:", abs_rel(np.asarray(deps), np.asarray(gt_deps)))

    pred_array = np.asarray(deps)
    gt_array = np.asarray(gt_deps)

    l1_error = np.abs(pred_array - gt_array)
    save_error_video_colormap(
        l1_error, osp.join(src, "debug", "depth_anything", "l1.mp4"), cmap="inferno"
    )
    l2_error = (pred_array - gt_array) ** 2
    save_error_video_colormap(
        l2_error, osp.join(src, "debug", "depth_anything", "l2.mp4"), cmap="inferno"
    )
    abs_rel_error = np.abs(pred_array - gt_array) / (gt_array + 1e-8)
    save_error_video_colormap(
        abs_rel_error,
        osp.join(src, "debug", "depth_anything", "abs_rel.mp4"),
        cmap="inferno",
    )

    dep_mask, _ = laplacian_filter_depth(deps)
    # dep_mask = dep_mask * (deps > depth_min) * (deps < depth_max)
    # dep = np.clip(dep, depth_min, depth_max)
    print("Abs Rel (masked):", abs_rel(np.asarray(deps), np.asarray(gt_deps), dep_mask))
