import imageio
from loguru import logger
import numpy as np
import torch
import os
import os.path as osp

from tqdm import tqdm
from torch_geometric.nn import knn

from monocular_cameras.backproject import backproject
from monocular_cameras.cameras import MonocularCameras
from motion_scaffold.dynamic_curves import lift_3d


@torch.no_grad()
def identify_fg_mask_by_nearest_curve(
    ws: str,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    cams: MonocularCameras,
    s_track: torch.Tensor,
    s_track_mask: torch.Tensor,
    d_track: torch.Tensor,
    d_track_mask: torch.Tensor,
):
    viz_dir = osp.join(ws, "viz", "mask")
    os.makedirs(viz_dir, exist_ok=True)
    time, height, width, channels = rgb.shape
    # get global anchor
    s_curve_xyz, _ = lift_3d(
        cams=cams, rgb=rgb, dep=dep, track=s_track, track_mask=s_track_mask
    )
    d_curve_xyz, _ = lift_3d(
        cams=cams, rgb=rgb, dep=dep, track=d_track, track_mask=d_track_mask
    )

    # only consider the valid case
    static_curve_mean = s_curve_xyz.sum(0, keepdim=True)
    # s_curve_xyz[:] = static_curve_mean
    np.savetxt(
        osp.join(viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
        static_curve_mean.reshape(-1, 3).cpu().numpy(),
        fmt="%.4f",
    )

    curve_xyz = torch.cat((s_curve_xyz, d_curve_xyz), dim=1)

    with torch.no_grad():
        fg_mask_list = []
        for query_tid in tqdm(range(time)):
            query_dep = dep[query_tid].clone()  # H,W
            query_xyz_cam = backproject(
                cams.get_homo_coordinate_map(),
                query_dep,
                rel_focal=cams.rel_focal,
                cxcy_ratio=cams.cxcy_ratio,
            )
            query_xyz_world = cams.trans_pts_to_world(query_tid, query_xyz_cam)  # H,W,3

            # find the nearest distance and acc sk weight
            # use all the curve at this position to id the fg and bg
            knn_idx = knn(
                curve_xyz[query_tid],
                query_xyz_world.reshape(-1, 3),
                k=1,
            )[1]  # [H*W, 1]

            fg_mask = torch.where(
                knn_idx > s_curve_xyz.shape[1],
                torch.ones((height * width)).to(knn_idx.device),
                torch.zeros((height * width)).to(knn_idx.device),
            ).reshape(height, width)
            fg_mask_list.append(fg_mask.cpu())
        fg_mask_list = torch.stack(fg_mask_list, 0)

    viz_rgb = rgb.clone().cpu()
    viz_rgb = viz_rgb * fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
        1 - fg_mask_list.float()[..., None]
    )
    imageio.v3.imwrite(
        osp.join(viz_dir, "foreground_gather_mask.mp4"),
        viz_rgb.to(torch.uint8).numpy(),
    )
    logger.info(
        f"foreground_mask saved at {osp.join(viz_dir, 'foreground_gather_mask.mp4')}"
    )

    fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool()
    return fg_mask_list
