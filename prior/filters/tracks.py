import torch
import logging
import numpy as np

from dataclasses import dataclass

@dataclass
class TrackFilterConfig:
    min_valid_cnt:int = 5


@torch.no_grad()
def filter_track(track:torch.Tensor, track_vis:torch.Tensor, dep_mask_list, cfg:TrackFilterConfig):
    # query valid depth and visible mask
    track_dep_mask, _ = gather_track_from_buffer(
        track[..., :2].long(), track_vis, dep_mask_list
    )
    track_mask = track_dep_mask.squeeze(-1) * track_vis
    # check whether a track is visible and has valid depth more more thant min_valid_cnt times
    valid_cnt = track_mask.sum(0)
    filter_track_mask = valid_cnt >= cfg.min_valid_cnt
    logging.info(
        f"Valid check: min_cnt={cfg.min_valid_cnt} {(~filter_track_mask).sum()} tracks are removed!"
    )
    track = track[:, filter_track_mask]
    track_mask = track_mask[:, filter_track_mask].clone()
    return track, track_mask

@torch.no_grad()
def gather_track_from_buffer(track: torch.Tensor, track_base_mask: torch.Tensor, buffer_list: torch.Tensor):
    # buffer_list: T, H, W, C
    if buffer_list.ndim == 3:
        buffer_list = buffer_list.unsqueeze(3)
    T, N = track_base_mask.shape
    C = buffer_list.shape[-1]
    
    # Initialize with nan
    ret = torch.full((T, N, C), float('nan'), dtype=buffer_list.dtype, device=track.device)
    ret_mask = torch.zeros_like(track_base_mask, dtype=torch.bool)
    
    for tid in range(T):
        _mask = track_base_mask[tid]
        if _mask.any():
            _uv_int = track[tid][_mask]
            _value = query_image_buffer_by_pix_int_coord(buffer_list[tid], _uv_int)
            # Ensure _value is a tensor
            if not isinstance(_value, torch.Tensor):
                _value = torch.tensor(_value, dtype=ret.dtype, device=ret.device)
            ret[tid, _mask] = _value
            ret_mask[tid, _mask] = True
    
    return ret, ret_mask

def query_image_buffer_by_pix_int_coord(buffer:torch.Tensor|np.ndarray, pixel_int_coordinate:np.ndarray|torch.Tensor):
    assert pixel_int_coordinate.ndim == 2 and pixel_int_coordinate.shape[-1] == 2
    assert (pixel_int_coordinate[..., 0] >= 0).all()
    assert (pixel_int_coordinate[..., 0] < buffer.shape[1]).all()
    assert (pixel_int_coordinate[..., 1] >= 0).all()
    assert (pixel_int_coordinate[..., 1] < buffer.shape[0]).all()
    # u is the col, v is the row
    col_id, row_id = pixel_int_coordinate[:, 0], pixel_int_coordinate[:, 1]
    H, W = buffer.shape[:2]
    index = col_id + row_id * W
    ret = buffer.reshape(H * W, *buffer.shape[2:])[index]
    if isinstance(ret, np.ndarray):
        ret = ret.copy()
    return ret