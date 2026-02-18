import torch


def __compute_delta_Rt_ji__(
    R_wi: torch.Tensor, t_wi: torch.Tensor, R_wj: torch.Tensor, t_wj: torch.Tensor
):
    # R: N,K,3,3; t: N,K,3
    # the stored node R,t are R_wi, t_wi
    # p_t=i_world = R_wi @ p_local + t_wi
    # p_local = R_wi.T @ (p_t=i_world - t_wi)
    # p_t=j_world = R_wj @ p_local + t_wj
    # p_t=j_world = R_wj @ R_wi.T @ (p_t=i_world - t_wi) + t_wj
    # p_t=j_world = (R_wj @ R_wi.T) @ p_t=i_world + t_wj - (R_wj @ R_wi.T) @ t_wi
    assert R_wi.ndim == 4 and R_wi.shape[2:] == (3, 3)
    assert t_wi.ndim == 3 and t_wi.shape[2] == 3
    assert R_wj.ndim == 4 and R_wj.shape[2:] == (3, 3)
    assert t_wj.ndim == 3 and t_wj.shape[2] == 3

    R_ji = torch.einsum("nsij,nskj->nsik", R_wj, R_wi)
    t_ji = t_wj - torch.einsum("nsij,nsj->nsi", R_ji, t_wi)
    return R_ji, t_ji
