from dataclasses import dataclass
from loguru import logger
import torch
from tqdm import tqdm
from utils3d.torch import matrix_to_quaternion

from monocular_cameras.cameras import MonocularCameras


@dataclass
class LeavesFetchConfig:
    n_attach: int | None = 30000  # number of leaves to attach if set to none attach all
    start_t = 0
    end_t: int | None = None
    subsample: int | None = None
    squeeze_normal_ratio = 1.0


@torch.no_grad()
def fetch_leaves_in_world_frame(
    cams: MonocularCameras,
    cfg: LeavesFetchConfig,
    input_mask_list: torch.Tensor,
    input_dep_list: torch.Tensor,
    input_rgb_list: torch.Tensor,
    t_list: list[int] | None = None,
):
    if cfg.end_t is None:
        end_t = cams.T
    else:
        end_t = cfg.end_t

    if cfg.subsample is not None and cfg.subsample > 1:
        logger.info(f"2D Subsample {cfg.subsample} for fetching ...")

    mu_list, quat_list, scale_list, rgb_list, time_index_list = [], [], [], [], []

    for t in tqdm(range(cfg.start_t, end_t) if t_list is None else t_list):
        mask2d = input_mask_list[t].bool()
        H, W = mask2d.shape
        if cfg.subsample is not None and cfg.subsample > 1:
            mask2d[:: cfg.subsample, :: cfg.subsample] = False

        dep_map = input_dep_list[t].clone()
        cam_pcl = cams.backproject(
            cams.get_homo_coordinate_map(H, W)[mask2d].clone(), dep_map[mask2d]
        )
        mu = cams.trans_pts_to_world(t, cam_pcl)
        rgb_map = input_rgb_list[t].clone()
        rgb = rgb_map[mask2d]
        K = cams.K(H, W)
        radius = (
            cam_pcl[:, -1] / (0.5 * K[0, 0] + 0.5 * K[1, 1]) * float(cfg.subsample or 1)
        )
        scale = torch.stack([radius / cfg.squeeze_normal_ratio, radius, radius], dim=-1)
        time_index = torch.ones_like(mu[:, 0]).long() * t
        quat = matrix_to_quaternion(torch.eye(3)[None].expand(len(radius), -1, -1))
        mu_list.append(mu.cpu())
        quat_list.append(quat.cpu())
        scale_list.append(scale.cpu())
        rgb_list.append(rgb.cpu())
        time_index_list.append(time_index.cpu())

    mu_all = torch.cat(mu_list, 0)
    quat_all = torch.cat(quat_list, 0)
    scale_all = torch.cat(scale_list, 0)
    rgb_all = torch.cat(rgb_list, 0)

    if cfg.n_attach is not None and cfg.n_attach > len(mu_all):
        choice = torch.randperm(len(mu_all))[: cfg.n_attach]
        logger.info(
            f"Fetching {cfg.n_attach / 1000.0:.3f}K out of {len(mu_all) / 1e6:.3}M pts"
        )
    else:
        choice = torch.arange(len(mu_all))
        logger.info(f"Fetching all of {len(mu_all) / 1000.0:.3f}K pts")

    # make gs5 param (mu, fr, s, o, sph) no rescaling
    mu_init = mu_all[choice].clone()
    q_init = quat_all[choice].clone()
    s_init = scale_all[choice].clone()
    o_init = torch.ones(len(choice), 1).to(mu_init)
    rgb_init = rgb_all[choice].clone()
    time_init = torch.cat(time_index_list, 0)[choice]
    torch.cuda.empty_cache()
    return (
        mu_init,
        q_init,
        s_init,
        o_init,
        rgb_init,
        time_init,
    )
