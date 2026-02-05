from dataclasses import dataclass
from typing import Literal
from loguru import logger
import torch

from prior.tracking.epi import analyze_track_epi


@dataclass
class TrackIdentificationConfig:
    epi_th = 1e-4
    static_cnt = 1
    dynamic_cnt = 1
    method: Literal["epi", "raft"] = "epi"


def identify_tracks(
    tracks: torch.Tensor, height: int, width: int, cfg: TrackIdentificationConfig
):
    if cfg.method == "raft":
        logger.info("Use pre-computed 2D epi error to mark static tracks")
        raise NotImplementedError(
            "Not implemented yet, need to collect the epi error from the pre-compute 2D epi from raft"
        )
    elif cfg.method == "epi":
        logger.info("Analyze the track epi to mark static tracks")
        F_list, epierr_list, _ = analyze_track_epi(
            make_continious_pair_list(tracks.shape[0]),
            tracks,
            torch.ones(tracks.shape[:2], dtype=torch.bool, device=tracks.device),
            H=height,
            W=width,
        )
    else:
        raise ValueError(f"Unknown track identification method: {cfg.method}")

    over_th_cnt = (epierr_list > cfg.epi_th).sum(0)
    dyn_mask = over_th_cnt >= cfg.dynamic_cnt
    sta_mask = over_th_cnt < cfg.static_cnt
    logger.warning(
        f"Identify {sta_mask.sum()} static, {dyn_mask.sum()} dynamic outof {len(sta_mask)} tracks"
    )
    return sta_mask, dyn_mask


def make_continious_pair_list(N) -> list[tuple[int, int]]:
    pair_list: list[tuple[int, int]] = []
    for i in range(N - 1):
        pair_list.append((i, i + 1))
    return pair_list
