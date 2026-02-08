from motion_scaffold import __compute_delta_Rt_ji__


def __DQB_warp__(
    sk_w,
    src_xyz,
    sk_src_node_xyz,
    sk_src_node_quat,
    sk_dst_node_xyz,
    sk_dst_node_quat,
    dyn_o,
    src_R=None,
):
    sk_R_tq, sk_t_tq = __compute_delta_Rt_ji__(
        R_wj=q2R(sk_dst_node_quat),
        t_wj=sk_dst_node_xyz,
        R_wi=q2R(sk_src_node_quat),
        t_wi=sk_src_node_xyz,
    )
    sk_dq_tq = Rt2dq(sk_R_tq, sk_t_tq)  # N,K,8
    # * Dual Quaternion skinning
    dq = torch.einsum("nki,nk->ni", sk_dq_tq, sk_w)  # N,8
    # use dyn mask to blend a unit dq into
    unit_dq = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0]).to(sk_w)[None].expand(len(dq), -1)
    dq = dq * dyn_o[:, None] + unit_dq * (1 - dyn_o)[:, None]
    with torch.no_grad():
        assert dq.max() < 1e6, f"{dq.max()}"
    dq = dq2unitdq(dq)
    R_tq, t_tq = dq2Rt(dq)  # N,3,3; N,3
    # * apply the transformation to the leaf
    mu_dst = torch.einsum("nij,nj->ni", R_tq, src_xyz) + t_tq
    if src_R is not None:
        fr_dst = torch.einsum("nij,njk->nik", R_tq, src_R)
    else:
        fr_dst = None
    return mu_dst, fr_dst
