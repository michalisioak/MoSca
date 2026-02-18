from loguru import logger
import torch

from motion_scaffold import MoSca, q2R


def compute_arap_loss(
    scf: MoSca,
    tids: torch.Tensor | None = None,
    temporal_diff_weight: list[float] = [0.75, 0.25],
    temporal_diff_shift: list[int] = [1, 4],
    # * used for only change the latest append frame during appending loop
    detach_tids_mask=None,
    reduce_type="mean",
    square=False,
):
    assert len(temporal_diff_weight) == len(temporal_diff_shift)
    if tids is None:
        tids = torch.arange(scf.T).to(scf.device)
    assert tids.max() <= scf.T - 1
    xyz = scf.node_xyz[tids]
    R_wi = q2R(scf.node_rotation[tids])
    topo_ind = scf.topo_knn_ind
    topo_w = scf.topo_knn_mask.float()  # N,K, binary mask
    topo_w = topo_w / (topo_w.sum(dim=-1, keepdim=True) + 1e-6)  # normalize
    local_coord = get_local_coord(xyz, topo_ind, R_wi)

    if detach_tids_mask is not None:
        detach_tids_mask = detach_tids_mask.float()
        local_coord = (
            local_coord.detach() * detach_tids_mask[:, None, None, None]
            + local_coord * (1 - detach_tids_mask)[:, None, None, None]
        )
    loss_coord, loss_len, _, _ = compute_arap(
        local_coord,
        topo_w,
        temporal_diff_shift,
        temporal_diff_weight,
        reduce=reduce_type,
        square=square,
    )

    # topo: speed up this
    if scf.mlevel_arap_flag:
        for level in range(len(scf.multilevel_arap_edge_list)):
            # ! in this case, self is from the larger set
            _local_coord = get_local_coord(
                xyz,
                scf.multilevel_arap_edge_list[level][:, 1:],
                R_wi,
                self_ind=scf.multilevel_arap_edge_list[level][:, :1],
                detach_nn=bool(scf.mlevel_detach_nn_flag.item()),
                detach_self=bool(scf.mlevel_detach_self_flag.item()),
            )  # T,N,1,3

            _loss_coord, _loss_len, _, _ = compute_arap(
                _local_coord,
                scf.multilevel_arap_topo_w[level][:, None],
                temporal_diff_shift,
                temporal_diff_weight,
                reduce=reduce_type,
                square=square,
            )
            loss_coord = loss_coord + _loss_coord * scf.mlevel_w_list[level]
            loss_len = loss_len + _loss_len * scf.mlevel_w_list[level]

    return loss_coord, loss_len


def compute_arap(
    local_coord: torch.Tensor,  # T,N,K,3
    topo_w: torch.Tensor,  # N,K
    temporal_diff_shift: list[int],
    temporal_diff_weight: list[float],
    reduce="mean",
    square=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local_coord_len = local_coord.norm(dim=-1, p=2)  # T,N,K
    # the coordinate should be similar
    # the metric should be similar
    loss_coord: torch.Tensor = torch.tensor(0.0).to(local_coord)
    loss_len: torch.Tensor = torch.tensor(0.0).to(local_coord)

    for shift, _w in zip(temporal_diff_shift, temporal_diff_weight):
        diff_coord: torch.Tensor = (local_coord[:-shift] - local_coord[shift:]).norm(
            dim=-1
        )
        if len(diff_coord) < 1:
            continue
        diff_len = (local_coord_len[:-shift] - local_coord_len[shift:]).abs()
        if square:
            logger.warning(
                "Use square loss, but initial exp shows non-square loss is better!"
            )
            diff_coord = diff_coord**2
            diff_len = diff_len**2
        diff_coord = (diff_coord * topo_w[None]).sum(-1)
        diff_len = (diff_len * topo_w[None]).sum(-1)
        if reduce == "sum":
            loss_coord: torch.Tensor = loss_coord + _w * diff_coord.sum()
            loss_len: torch.Tensor = loss_len + _w * diff_len.sum()
        elif reduce == "mean":
            loss_coord: torch.Tensor = loss_coord + _w * diff_coord.mean()
            loss_len: torch.Tensor = loss_len + _w * diff_len.mean()
        else:
            raise NotImplementedError()
    loss_coord_global: torch.Tensor = (local_coord.std(0) * topo_w[..., None]).sum()
    loss_len_global: torch.Tensor = (local_coord_len.std(0) * topo_w).sum()
    assert not torch.isnan(loss_coord) and not torch.isnan(loss_len)
    return loss_coord, loss_len, loss_coord_global, loss_len_global


def get_local_coord(
    xyz: torch.Tensor,
    topo_ind,
    R_wi,
    self_ind: torch.Tensor | None = None,
    detach_self=False,
    detach_nn=False,
):
    assert not (detach_self and detach_nn), "detach_self and detach_nn are exclusive"
    # * self will be expressed in nn coordinate frame
    # xyz: T,N,3; topo_ind: N,K; R_wi: T,N,3,3
    nn_xyz = xyz[:, topo_ind, :]
    nn_R_wi = R_wi[:, topo_ind, :]
    if self_ind is None:
        self_xyz = xyz[:, :, None]  # T,N,1,3
    else:
        assert self_ind.shape == topo_ind.shape
        self_xyz = xyz[:, self_ind, :]
    if detach_self:
        self_xyz = self_xyz.detach()
    if detach_nn:
        nn_xyz = nn_xyz.detach()
    local_coord = torch.einsum(
        "tnkji,tnkj->tnki", nn_R_wi, self_xyz - nn_xyz
    )  # T,N,K,3
    return local_coord
