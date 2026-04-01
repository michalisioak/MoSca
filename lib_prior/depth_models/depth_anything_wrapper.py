import logging
import os
import os.path as osp
import sys
import cv2
from numpy import ndarray
import numpy as np

sys.path.append(osp.abspath(osp.dirname(__file__)))

import os, os.path as osp

import torch

from depth_anything_3.api import DepthAnything3
from depth_utils import viz_depth_list, save_depth_list


def get_depth_anything_model(device="cuda"):
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
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
    frame_window_size=40,
    overlap=0.8,
    intrs: ndarray | None = None,
    extrs: ndarray | None = None,
):
    print("Depth-Anything processing...")
    os.makedirs(dst, exist_ok=True)
    print(f"Processing {img_list.shape} images...")
    T, H, W, C = img_list.shape
    if frame_window_size <= 0:
        frame_window_size = T
    dep_list = []
    curr = 0
    overlap_frames = int(frame_window_size * overlap)
    # intrs = []
    while curr + overlap_frames < T:
        print(f"Processing frames {curr} to {min(curr + frame_window_size, T)}...")
        batched_images = list(img_list[curr : min(curr + frame_window_size, T)])
        batched_extrs = (
            list(extrs[curr : min(curr + frame_window_size, T)])
            if extrs is not None
            else None
        )
        pred = model.inference(
            batched_images,
            ref_view_strategy="middle",
            process_res=max(H, W),
            intrinsics=np.asarray([intrs] * len(batched_images))
            if intrs is not None
            else None,
            extrinsics=np.asarray(batched_extrs),
            # intrinsics=np.array(
            #     intrs[curr - frame_window_size : min(curr, T - frame_window_size)]
            # )
            # if curr >= frame_window_size
            # else None,
        )

        local_dep = pred.depth
        local_intrs = pred.intrinsics
        assert local_intrs is not None
        if curr == 0:
            dep_list.extend(local_dep)
            # intrs.extend(list(local_intrs))
        else:
            for i in range(overlap_frames):
                alpha = (i + 1) / (overlap_frames + 1)
                blended = (
                    dep_list[-overlap_frames + i] * (1 - alpha) + local_dep[i] * alpha
                )
                dep_list[-overlap_frames + i] = blended
            dep_list.extend(local_dep[overlap_frames:])
            # intrs.extend(list(local_intrs[overlap_frames:]))

        curr += frame_window_size - overlap_frames

    print(f"Saving depth maps... for {len(dep_list)} frames")
    save_depth_list(dep_list, fn_list, dst, invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    torch.cuda.empty_cache()
    return dep_list


if __name__ == "__main__":
    device = "cuda"
    src = "./demo/train/images"

    import imageio

    src = osp.expanduser(src)
    fns = os.listdir(src)
    fns.sort()
    fn_list = [fn for fn in fns if fn.endswith(".jpg") or fn.endswith(".png")]

    model = get_depth_anything_model(device=device)
    depth_anything_proccess_folder(
        model,
        fn_list=fn_list,
        img_list=np.asarray([imageio.imread(src + "/" + fn) for fn in fn_list]),
        dst=osp.join(src, "../debug/depth_anything"),
    )
