from loguru import logger
import torch
from tqdm import tqdm

from monocular_cameras.cameras import MonocularCameras
from utils.buffer import prepare_track_homo_dep_rgb_buffers
from .utils import get_world_points


def __int2homo_coord__(track_uv, H, W):
    # the short side is [-1,1]
    H, W = float(H), float(W)
    L = min(H, W)
    homo_x = (track_uv[..., 0] + 0.5) / L * 2 - (W / L)
    homo_y = (track_uv[..., 1] + 0.5) / L * 2 - (H / L)
    homo = torch.stack([homo_x, homo_y], -1)
    return homo


@torch.no_grad()
def line_segment_init(track_mask, point_ref):
    logger.info("Naive Line Segment Init")
    # ! this function is a bad init, but this doesn't matter, later will directly optimize the curve

    tracl_mask_valid_cnt = track_mask.sum(0)
    working_mask = tracl_mask_valid_cnt > 0
    logger.info(f"Line Segment Init, invalid curve cnt={(~working_mask).sum()}")

    working_point_ref = point_ref.detach().clone()[:, working_mask]
    working_track_mask = track_mask[:, working_mask]

    T, N = track_mask.shape
    # point_ref # T,N,3
    # scan the T, for each empty slot, identify the right ends, and compute linear interpolation, if there is only one side, stay at the same position, if two end are empty, assert error, there shouldn't be an empty noodle!
    inverse_muti = torch.Tensor([i + 1 for i in range(T)][::-1]).to(working_point_ref)
    for t in tqdm(range(T)):
        to_fill_mask = ~working_track_mask[t]
        if not to_fill_mask.any():
            continue  # skip this time if everything is filled
        # identify the left and right nearest valid side

        if t == T - 1:  # if right end, use the previous one
            value = working_point_ref[t - 1, to_fill_mask].clone()
        else:
            # identify the right end, the left end must be filled in already
            to_fill_valid_curve = working_track_mask[t + 1 :, to_fill_mask]  # T,M
            # find the left most True slot
            to_fill_valid_curve = (
                to_fill_valid_curve.float() * inverse_muti[t + 1 :, None]
            )
            max_value, max_ind = to_fill_valid_curve.max(dim=0)
            # for no right mask case, use the left
            select_from = working_point_ref[t + 1 :, to_fill_mask]
            valid_right_end = torch.gather(
                select_from, 0, max_ind[None, :, None].expand(-1, -1, 3)
            )[0, max_value > 0]  # valid when max_value > 0
            if t == 0:
                assert len(valid_right_end) == to_fill_valid_curve.shape[1], (
                    "empty noodle!"
                )
                value = valid_right_end
            else:
                # must have a left end
                value = working_point_ref[t - 1, to_fill_mask].clone()
                valid_left_end = value[max_value > 0]
                delta_t = (
                    max_ind[max_value > 0] + 2
                )  # left valid, current, [0] in the max_ind
                delta_x = valid_right_end - valid_left_end
                inc = 1.0 * delta_x / delta_t[:, None]
                value[max_value > 0] = valid_left_end + inc
        working_point_ref[t, to_fill_mask] = value.clone()
    # np.savetxt("./debug/line_segment_init.xyz", point_ref.reshape(-1, 3).cpu().numpy())
    ret = point_ref.clone()
    ret[:, working_mask] = working_point_ref
    return ret.detach().clone()


def lift_3d(
    cams: MonocularCameras,
    rgb: torch.Tensor,
    dep: torch.Tensor,
    track: torch.Tensor,
    track_mask: torch.Tensor,
    # return_all_curves=False,
    # # filter of 2D tracks to avoid the fg-bg error track flickering
    # refilter_2d_track_flag=True,
    # refilter_2d_track_only_mask=False,  # if set true won't remove any curve, just mark the outlier as invalid
    # refilter_min_valid_cnt=2,
    # refilter_o3d_nb_neighbors=16,
    # refilter_o3d_std_ratio=5.0,
    # refilter_shaking_th=0.2,
    # refilter_remove_shaking_curve=False,  # if true, any curve with shaking will be totally removed
    # refilter_spatracker_consistency_th=0.2,
    # #
    # spatracker_original_curve=False,
    # # additional mask, this is for the semantic label consistency mask
    # fg_additional_mask=None,
    # # spatracker 3D curve also have a choice to use line init
    # enforce_line_init=False,
    # # subsample t list
    # t_list=None,
    # # safe cfgs
    # min_num_curves=0,
):
    time, height, width, channels = rgb.shape
    if track.shape[-1] == 3:
        logger.info("SpaT mode, direct use 3D Track")
        # spa tracker model
        # manually homo list
        homo_list = __int2homo_coord__(track[..., :2], height, width)
        dep_list = track[..., -1]
        curve_xyz = get_world_points(homo_list, dep_list, cams)

        _, _, rgb_list = prepare_track_homo_dep_rgb_buffers(
            rgb, dep, track[..., :2], track_mask
        )

    else:
        logger.info("2D track mode, use line segment to fill")
        homo_list, dep_list, rgb_list = prepare_track_homo_dep_rgb_buffers(
            rgb, dep, track[..., :2], track_mask
        )
        curve_xyz = line_segment_init(
            track_mask, get_world_points(homo_list, dep_list, cams).clone()
        )

    return curve_xyz, rgb_list
