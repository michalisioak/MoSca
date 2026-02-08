def fast_warp(
    self,
    target_tid,
    # all below are baked
    sk_ind,
    sk_w,
    sk_ref_node_xyz,
    sk_ref_node_quat,
    dyn_o,
    query_xyz,
    query_dir,
):
    # query_xyz: (N, 3) in live world frame, time: N,
    # query_dir: (N,3,C), attach_node_ind: N, must specify outside which curve is the nearest, so the topology is decided there
    # note, the query_tid and target_tid can be different for each query
    sk_dst_node_xyz = self.node_xyz[target_tid][sk_ind]
    sk_dst_node_quat = self._node_rotation[target_tid][sk_ind]
    ret_xyz, ret_dir = self.__BLEND_FUNC__(
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
