import torch

from monocular_cameras.backproject import backproject
from monocular_cameras.cameras import MonocularCameras


def get_world_points(
    homo_list: torch.Tensor,
    dep_list: torch.Tensor,
    cams: MonocularCameras,
    cam_t_list=None,
):
    T, M = dep_list.shape
    if cam_t_list is None:
        cam_t_list = torch.arange(T).to(homo_list.device)
    point_cam = backproject(
        homo_list.reshape(-1, 2),
        dep_list.reshape(-1),
        rel_focal=cams.rel_focal,
        cxcy_ratio=cams.cxcy_ratio,
    )
    point_cam = point_cam.reshape(T, M, 3)
    R_wc, t_wc = cams.Rt_wc_list()
    R_wc, t_wc = R_wc[cam_t_list], t_wc[cam_t_list]
    point_world = torch.einsum("tij,tmj->tmi", R_wc, point_cam) + t_wc[:, None]
    return point_world
