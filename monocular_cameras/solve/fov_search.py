from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import os.path as osp


@dataclass
class FovSearchConfig:
    fallback_fov: float = 40.0
    search_N: int = 100
    search_start: float = 30.0
    search_end: float = 100.0


@torch.no_grad()
def find_initinsic(
    pair_list,
    pair_mask_list,
    height: int,
    width: int,
    track,
    dep_list,
    depth_decay_th: float,
    depth_decay_sigma: float,
    cfg: FovSearchConfig,
    ws: str,
):
    logger.info("Start FOV search...")
    # * assume fov is known, solve the optimal s,R,t between view pair and form an energy under such case
    # convert the [0,W], [0,H] to aspect un-distorted homo_list
    homo_list = track2undistroed_homo(track, height, width)

    e_list, fov_list, R_list, t_list = [], [], [], []
    search_candidates = np.linspace(cfg.search_start, cfg.search_end, num=cfg.search_N)[
        1:-1
    ]
    for fov in tqdm(search_candidates):
        E, E_i, s_ij, R_ij, t_ij = compute_graph_energy(
            fov,
            pair_list,
            pair_mask_list,
            homo_list,
            dep_list,
            depth_decay_th=depth_decay_th,
            depth_decay_sigma=depth_decay_sigma,
        )
        e_list.append(E.item())
        fov_list.append(fov)
        R_list.append(R_ij)
        t_list.append(t_ij)
    e_list = np.array(e_list)
    best_ind = e_list.argmin()
    optimial_fov = fov_list[best_ind]
    optima_R_ij = R_list[best_ind]
    optima_t_ij = t_list[best_ind]
    # ! detect monotonic case and use fallback fov if no optimal is found
    if (e_list[1:] >= e_list[:-1]).all() or (e_list[1:] <= e_list[:-1]).all():
        logger.warning(
            f"FOV search mono case encountered, fall back to FOV={cfg.fallback_fov}"
        )
        optimial_fov = cfg.fallback_fov

    plt.figure(figsize=(10, 3))
    plt.plot(fov_list, e_list)
    plt.plot([optimial_fov, optimial_fov], [min(e_list), max(e_list)], "r--")
    plt.plot([optimial_fov], [min(e_list)], "o")
    plt.title(
        f"FOV Linear Search Best={optimial_fov:.3f} with energy {min(e_list):.6f}"
    )
    plt.xlabel("fov")
    plt.ylabel("ReprojEnergy")
    plt.tight_layout()
    plt.savefig(osp.join(ws, "camera", "moca", "fov_search.jpg"))
    plt.close()
    logger.info(f"FOV search done, find FOV={optimial_fov:.3f} deg")

    return optimial_fov, optima_R_ij, optima_t_ij
