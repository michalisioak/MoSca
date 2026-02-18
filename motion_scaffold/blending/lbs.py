from typing import Optional
import torch

from utils3d.torch import matrix_to_quaternion

from motion_scaffold.blending import __compute_delta_Rt_ji__
from motion_scaffold.dualquat import q2R


def __LBS_warp__(
    sk_w,
    src_xyz,
    sk_src_node_xyz,
    sk_src_node_quat,
    sk_dst_node_xyz,
    sk_dst_node_quat,
    dyn_o,
    src_R=None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    sk_R_tq, sk_t_tq = __compute_delta_Rt_ji__(
        R_wj=q2R(sk_dst_node_quat),
        t_wj=sk_dst_node_xyz,
        R_wi=q2R(sk_src_node_quat),
        t_wi=sk_src_node_xyz,
    )
    sk_quat_tq = matrix_to_quaternion(sk_R_tq)
    quat_tq = torch.einsum("nki,nk->ni", sk_quat_tq, sk_w)  # N,4
    t_tq = torch.einsum("nki,nk->ni", sk_t_tq, sk_w)  # N,3
    q_unit = torch.Tensor([1, 0, 0, 0]).to(sk_w)[None]
    quat_tq = quat_tq * dyn_o[:, None] + (1 - dyn_o)[:, None] * q_unit
    t_tq = t_tq * dyn_o[:, None]
    R_tq = q2R(quat_tq)
    mu_dst = torch.einsum("nij,nj->ni", R_tq, src_xyz) + t_tq
    if src_R is not None:
        fr_dst = torch.einsum("nij,njk->nik", R_tq, src_R)
    else:
        fr_dst = None
    return mu_dst, fr_dst
