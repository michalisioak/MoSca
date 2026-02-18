from loguru import logger
import torch
from tqdm import tqdm

from lib_mosca.mosca import MoSca
from monocular_cameras.cameras import MonocularCameras
from scene.cfg import GaussianSplattingConfig
from scene.dynamic.model import DynamicScene
from scene.fetch import LeavesFetchConfig, fetch_leaves_in_world_frame


@torch.no_grad()
def get_dynamic_model(
    rgb: torch.Tensor,
    dep: torch.Tensor,
    cams: MonocularCameras,
    scf: MoSca,
    fetch_cfg: LeavesFetchConfig,
    gather_mask: torch.Tensor,
    gs_cfg: GaussianSplattingConfig,
    radius_init_factor: float,
    opacity_init_factor: float,
):
    collect_t_list = torch.arange(0, s2d.T, 1)
    logger.info(f"Collect GS at t={collect_t_list}")
    input_mask_list = s2d.dyn_mask * s2d.dep_mask
    if additional_mask is not None:
        assert additional_mask.shape == s2d.dep_mask.shape
        input_mask_list = input_mask_list * additional_mask
    mu_init, q_init, s_init, o_init, rgb_init, time_init = fetch_leaves_in_world_frame(
        dep=dep, rgb=rgb, cams=cams, cfg=fetch_cfg, gather_mask=gather_mask
    )
    # * Reset SCF topo th!
    if topo_th_ratio is not None:
        old_th_ratio = scf.topo_th_ratio
        scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
        logging.info(
            f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
        )

    # * Init the scf
    d_model: DynamicScene = DynamicScene(
        scf=scf,
        max_scale=radius_max,
        min_scale=0.0,
        max_sph_order=max_sph_order,
        device=device,
        leaf_local_flag=leaf_local_flag,
        dyn_o_flag=dyn_o_flag,
        nn_fusion=nn_fusion,
        max_node_num=max_node_num,
    )

    # * Init the leaves
    optimizer = torch.optim.Adam(d_model.get_optimizable_list())
    unique_tid = time_init.unique()
    logger.info("Attach to Dynamic Scaffold ...")
    for tid in tqdm(unique_tid):
        t_mask = time_init == tid
        d_model.append_new_gs(
            optimizer,
            tid=tid,
            mu_w=mu_init[t_mask],
            quat_w=q_init[t_mask],
            scales=s_init[t_mask] * radius_init_factor,
            opacity=o_init[t_mask] * opacity_init_factor,
            rgb=rgb_init[t_mask],
        )
    d_model.scf.update_topology()
    d_model.summary()

    return d_model
