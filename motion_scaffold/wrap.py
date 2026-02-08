import torch

from motion_scaffold import MoSca


def warp(
    scf: MoSca,
    attach_node_ind,
    query_xyz,
    query_tid,
    target_tid,
    query_dir=None,
    skinning_w_corr=None,
    dyn_o_flag=False,
):
    # query_xyz: (N, 3) in live world frame, time: N,
    # query_dir: (N,3,C), attach_node_ind: N, must specify outside which curve is the nearest, so the topology is decided there
    # note, the query_tid and target_tid can be different for each query

    # * check
    if isinstance(query_tid, int) or query_tid.ndim == 0:
        query_tid = torch.ones_like(query_xyz[:, 0]).long() * query_tid
    N = len(query_tid)
    assert len(query_xyz) == N and query_xyz.shape == (N, 3)
    if query_dir is not None:
        assert query_dir.shape[:2] == (N, 3) and query_dir.ndim == 3
    if isinstance(target_tid, int) or target_tid.ndim == 0:
        target_tid = target_tid * torch.ones_like(query_tid)

    # * identify skinning weights
    sk_ind, sk_w, sk_w_sum, sk_ref_node_xyz, sk_ref_node_quat = (
        scf.get_skinning_weights(
            query_xyz=query_xyz,
            query_t=query_tid,
            attach_ind=attach_node_ind,
            skinning_weight_correction=skinning_w_corr,
        )
    )

    # * blending
    sk_dst_node_xyz, sk_dst_node_quat = scf.get_async_knns(target_tid, sk_ind)
    if dyn_o_flag:
        dyn_o = torch.clamp(sk_w_sum, min=0.0, max=1.0)
    else:
        dyn_o = torch.ones_like(sk_w_sum) * (1.0 - DQ_EPS)
    ret_xyz, ret_dir = scf.__BLEND_FUNC__(
        sk_w=sk_w,
        src_xyz=query_xyz,
        src_R=query_dir,
        sk_src_node_xyz=sk_ref_node_xyz,
        sk_src_node_quat=sk_ref_node_quat,
        sk_dst_node_xyz=sk_dst_node_xyz,
        sk_dst_node_quat=sk_dst_node_quat,
        dyn_o=dyn_o,
    )
    return ret_xyz, ret_dir
