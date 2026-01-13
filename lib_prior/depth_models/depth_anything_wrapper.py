import os
import  os.path as osp
import sys
import cv2
from numpy import ndarray
import numpy as np

sys.path.append(osp.abspath(osp.dirname(__file__)))

import os, os.path as osp

from depth_anything_3.api import DepthAnything3
from depth_utils import viz_depth_list, save_depth_list

def get_depth_anything_model(device = "cuda"):
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model.to(device=device)
    model.eval()
    return model

def depth_anything_proccess_folder(
    model: DepthAnything3,
    img_list: ndarray,
    fn_list: list[str],
    dst:str,
    invalid_mask_list=None,
):
    print("Depth-Anything processing...")
    os.makedirs(dst, exist_ok=True)
    T, H, W, C = img_list.shape
    dep_list_unscaled = model.inference(list(img_list), ref_view_strategy="middle",process_res=max(H,W),).depth
    dep_list = []
    for dep in dep_list_unscaled:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    dep_list = np.asarray(dep_list)
    save_depth_list(dep_list,fn_list,dst,invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    return

if __name__ == "__main__":
    device = "cuda"
    src = "./demo/train/images"

    src = osp.expanduser(src)
    fns = os.listdir(src)
    fns.sort()
    fn_list =  [fn for fn in fns if fn.endswith(".jpg") or fn.endswith(".png")]

    model = get_depth_anything_model(device=device)
    depth_anything_proccess_folder(
        model,
        fn_list=fn_list,
        img_list=np.asarray([src + "/" + fn for fn in fn_list]),
        dst=osp.join(src, "../debug/depth_anything"),
    )