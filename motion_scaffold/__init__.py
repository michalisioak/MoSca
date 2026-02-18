# Stand alone warping field with MoSca
from dataclasses import dataclass, field
import time
from typing import Literal
import torch
from torch import nn
from loguru import logger
from torch_geometric.nn import knn_graph, knn
from tqdm import tqdm

from motion_scaffold.blending.dqb import __DQB_warp__
from motion_scaffold.blending.lbs import __LBS_warp__
from motion_scaffold.gs_optim_helper import cat_tensors_to_optimizer, prune_optimizer

from .dualquat import Rt2dq, dq2unitdq, dq2Rt, q2R
from utils3d.torch import (
    matrix_to_quaternion,
)


@dataclass
class MoScaConfig:
    topo_chunk_size = 65536 // 8
    dq_eps = 0.001  # 0.00001
    spatial_unit_factor: float = 1  # ! important to tune
    spatial_unit_hard_set: float | None = None
    skinning_k: int = 16
    topo_dist_top_k: int = 1
    sigma_init_ratio = 5.0  # independent to the max sigma
    sigma_max_ratio = 10.0
    topo_dist_top_k = 3
    topo_th_ratio = 5.0
    topo_sample_T = 100
    skinning_method: Literal["dqb", "lbs"] = "dqb"
    mlevel_list: list[int] = field(default_factory=lambda: [1, 7, 15])
    mlevel_k_list: list[int] = field(default_factory=lambda: [16, 8, 8])
    mlevel_w_list: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    mlevel_detach_nn_flag = True  # ! this should be False but due to the old code behavior, set default to True to align with the submission version.
    mlevel_detach_self_flag = False
    w_corr_maintain_sum_flag = False
    min_node_num = 32
    max_node_num = 16384
    mlevel_arap_flag = True
    break_topo_between_group = (
        True  # by default don't construct topology edges between two groups
    )


class MoSca(nn.Module):
    def __init__(
        self,
        node_xyz,  # T,M,3
        node_certain,  # T,M # if valid (visible, inside depth) = 1
        cfg: MoScaConfig,
        node_grouping=None,  # M # ! warning edges can only be built with in a group, usually in real case, there must be edges across different object, so, use a dummy group here
        node_sigma_logit: torch.Tensor | None = None,
        t_list=None,
    ):
        # T,N,3; T,N,4
        super().__init__()
        self.cfg = cfg

        # * xyz
        if node_grouping is None:
            node_grouping = torch.zeros_like(node_certain[0], dtype=torch.int)  # M

        self.node_xyz = nn.Parameter(node_xyz)
        self.node_certain = nn.Buffer(node_certain)  # if certain True
        self.node_grouping = nn.Buffer(node_grouping)
        self.unique_grouping = nn.Buffer(torch.unique(self.node_grouping))

        # * compute topology with grouping
        # first identify the spatial unit
        if cfg.spatial_unit_hard_set is None or cfg.spatial_unit_hard_set <= 0.0:
            logger.info(
                f"Auto set spatial unit with factor={cfg.spatial_unit_factor} from curve median dist"
            )
            spatial_unit = (
                cfg.spatial_unit_factor
                * __identify_spatial_unit_from_curves__(
                    self.node_xyz, self.node_certain, K=cfg.skinning_k
                )
            )
            logger.info(f"Auto set spatial unit with factor={spatial_unit}")
        else:
            logger.info(
                f"Hard set spatial unit with factor={cfg.spatial_unit_hard_set}"
            )
            spatial_unit = cfg.spatial_unit_hard_set
        self.spatial_unit = nn.Buffer(torch.tensor(spatial_unit))

        self.skinning_k = nn.Buffer(torch.tensor(cfg.skinning_k).long())
        self.topo_dist_top_k = nn.Buffer(torch.tensor(cfg.topo_dist_top_k).long())
        self.topo_sample_T = nn.Buffer(torch.tensor(cfg.topo_sample_T).long())
        self.topo_th_ratio = nn.Buffer(torch.tensor(cfg.topo_th_ratio))
        self.fixed_topology_flag = nn.Buffer(torch.tensor(False))

        self.mlevel_arap_flag = cfg.mlevel_arap_flag
        if self.mlevel_arap_flag:
            self.mlevel_list = cfg.mlevel_list
            self.mlevel_k_list = cfg.mlevel_k_list
            self.mlevel_w_list = cfg.mlevel_w_list
            logger.info(
                f"Set MoSca with multi-level arap topo reg with level-list={self.mlevel_list}, k-list={self.mlevel_k_list}, w-list={self.mlevel_w_list}"
            )

        self.mlevel_detach_nn_flag = nn.Buffer(torch.tensor(cfg.mlevel_detach_nn_flag))
        self.mlevel_detach_self_flag = nn.Buffer(
            torch.tensor(cfg.mlevel_detach_self_flag)
        )
        self.topo_knn_ind = nn.Buffer(torch.zeros(0).long())
        self.topo_knn_mask = nn.Buffer(torch.zeros(0).bool())
        self.break_topo_between_group = nn.Buffer(cfg.break_topo_between_group)
        logger.info(f"mosca break topo between group: {cfg.break_topo_between_group}")
        self.update_topology(curve_mask=self.node_certain)

        # * compute optimal frame and init the rotation
        self.compute_rotation_from_xyz()

        # * RBF interpolation
        self.max_sigma = nn.Buffer(torch.tensor(cfg.sigma_max_ratio * spatial_unit))
        self.init_sigma = nn.Buffer(torch.tensor(cfg.sigma_init_ratio * spatial_unit))
        if node_sigma_logit is None:
            node_sigma_logit = self.sig_invact(torch.ones(self.M) * self.init_sigma)
        else:
            assert len(node_sigma_logit) == self.M
            assert node_sigma_logit.ndim <= 2
        if node_sigma_logit.ndim == 1:
            node_sigma_logit = node_sigma_logit[:, None]
        self.node_sigma_logit = nn.Parameter(node_sigma_logit)  # M,1

        # * Skinning config
        if cfg.skinning_method == "dqb":
            self.blending_func = __DQB_warp__
        elif cfg.skinning_method == "lbs":
            self.blending_func = __LBS_warp__
        else:
            raise ValueError(f"Not valid skinning method: {cfg.skinning_method}")

        self.w_corr_maintain_sum_flag = nn.Buffer(cfg.w_corr_maintain_sum_flag)
        self.t_list = nn.Buffer(torch.arange(self.T).long())
        if t_list is None:
            t_list = torch.arange(self.T).long()
            logger.info("[t_list] is not provided and set to default")
        else:
            t_list = torch.as_tensor(t_list).long()
            logger.info(f"[t_list] is provided: {t_list[:3]} ... {t_list[-3:]}")
        self.t_list = nn.Buffer(t_list)
        self.min_node_num = cfg.min_node_num
        self.max_node_num = cfg.max_node_num
        assert self.M >= self.min_node_num, (
            f"Node num {self.M} is less than {self.min_node_num}"
        )
        self.summary()

    @torch.no_grad()
    def compute_rotation_from_xyz(self):
        _node_quat = self._compute_R_from_xyz__()
        if hasattr(self, "_node_rotation"):
            self.node_rotation.data = _node_quat
        else:
            self.node_rotation = nn.Parameter(_node_quat)
        torch.cuda.empty_cache()

    @property
    def M(self):
        return self.node_xyz.shape[1]

    @property
    def T(self):
        return self.node_xyz.shape[0]

    def summary(self):
        logger.info(
            f"MoSca Summary: T={self.T}; M={self.M}; K={self.skinning_k}; spatial-unit={self.spatial_unit}"
        )

        logger.info(
            f"MoSca Summary[1]: T={self.T}; M={self.M}; K={self.skinning_k}; Multi-level={self.mlevel_list}"
        )
        logger.info(
            f"MoSca Summary[2]: curve-dist-K={self.topo_dist_top_k}; spatial-unit={self.spatial_unit}; topo-th-ratio={self.topo_th_ratio}"
        )
        logger.info(
            f"MoSca Summary[3]: SK-method={self.blending_method}; fixed-topology={self.fixed_topology_flag}; sigma-max={self.max_sigma}; sigma-init={self.init_sigma}"
        )

    def sig_act(self, x):
        return self.max_sigma * torch.sigmoid(x)

    def sig_invact(self, x):
        return torch.logit(torch.clamp(x / self.max_sigma, 1e-6, 1.0 - 1e-6))

    @property
    def node_sigma(self):
        return self.sig_act(self.node_sigma_logit)

    @torch.no_grad()
    def decremental_topology(
        self,
        node_prune_mask,  # if prune, mark as Ture
        verbose=False,
        multilevel_update_flag=True,
    ):
        ind_convert = torch.ones_like(node_prune_mask).long() * -1
        ind_convert[~node_prune_mask] = torch.arange(self.M, device=self.device)

        start_t = time.time()
        assert hasattr(self, "_D_topo")
        assert self.D_topo.shape[1] > self.M, "No need to decrement!"
        old_M = self.D_topo.shape[1]
        logger.info(
            f"Decremental topology from {old_M} to {self.M} (specially handle the knn)"
        )

        # update the D_topo
        self.D_topo = _compute_curve_topo_dist_(
            curve_xyz=self.node_xyz,
            curve_mask=None,
            top_k=int(self.topo_dist_top_k.item()),
            max_subsample_T=int(self.topo_sample_T.item()),
        )
        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)

        # manually update the knn ind and mask, for fixed topology update
        can_change_mask = node_prune_mask[self.topo_knn_ind[~node_prune_mask]]
        new_topo_knn_ind = ind_convert[self.topo_knn_ind[~node_prune_mask]]
        assert (can_change_mask == (new_topo_knn_ind == -1)).all()
        new_topo_knn_mask = self.topo_knn_mask[~node_prune_mask]
        logger.info(
            f"During topo decremental, only {can_change_mask.sum()} slot of knn ind and mask are changed"
        )

        # only change the can change place with a naive for loop
        topo_th = self.topo_th_ratio * self.spatial_unit
        any_change_id = torch.arange(len(can_change_mask))[
            can_change_mask.any(dim=1).cpu()
        ]
        for node_id in tqdm(any_change_id.cpu()):
            d = self.D_topo[node_id].clone()
            change_mask = can_change_mask[node_id]
            unchanged_ind = new_topo_knn_ind[node_id][~change_mask]
            d[unchanged_ind] = d.max()
            new_nn_id = d.topk(int(self.skinning_k.item()), largest=False).indices
            new_topo_knn_ind[node_id][change_mask] = new_nn_id[: change_mask.sum()]
            new_topo_knn_mask[node_id][change_mask] = d[
                new_nn_id[: change_mask.sum()]
            ] < (topo_th)
        self.topo_knn_ind = new_topo_knn_ind.clone()
        self.topo_knn_mask = new_topo_knn_mask.clone()
        if verbose:
            logger.info(f"Topology updated in {time.time() - start_t:.2f}s")
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def incremental_topology(
        self,
        verbose=False,
        multilevel_update_flag=True,
    ):
        # ! special func for error grow
        start_t = time.time()
        assert hasattr(self, "_D_topo")
        assert self.D_topo.shape[1] < self.M, "No need to increment!"
        old_M = self.D_topo.shape[1]
        logger.info(
            f"Incremental topology from {old_M} to {self.M} (not changing old {old_M})"
        )

        # update the D_topo
        old_M = len(self.D_topo)
        append_M = self.M - old_M
        new_D = None
        if append_M > 0:
            bottom = __query_distance_to_curve__(
                q_curve_xyz=self.node_xyz[:, old_M:],
                b_curve_xyz=self.node_xyz[:, :old_M],
                top_k=int(self.topo_dist_top_k.item()),
                max_subsample_T=int(self.topo_sample_T.item()),
                chunk=self.cfg.topo_chunk_size,
            )
            square = __query_distance_to_curve__(
                q_curve_xyz=self.node_xyz[:, old_M:],
                b_curve_xyz=self.node_xyz[:, old_M:],
                top_k=int(self.topo_dist_top_k.item()),
                max_subsample_T=int(self.topo_sample_T.item()),
                chunk=self.cfg.topo_chunk_size,
            )
            # all diag elements of square should be huge 1e10
            square = square + torch.eye(append_M).to(square) * 1e10

            new_D1 = torch.cat([self.D_topo, bottom.T], 1)
            new_D2 = torch.cat([bottom, square], 1)
            new_D = torch.cat([new_D1, new_D2], 0)
            self.D_topo = new_D

        # update the knn and mask
        assert len(self.topo_knn_ind) == old_M
        assert new_D is not None
        new_topo_dist, new_topo_knn_ind = __compute_topo_ind_from_dist__(
            new_D[old_M:], self.skinning_k - 1
        )
        new_self_ind = torch.arange(self.M).to(self.device)[old_M:]
        topo_ind = torch.cat([new_self_ind[:, None], new_topo_knn_ind], dim=1)
        topo_th = self.topo_th_ratio * self.spatial_unit
        topo_mask = new_topo_dist < topo_th
        topo_mask = torch.cat([torch.ones_like(topo_mask[:, :1]), topo_mask], 1)
        self.topo_knn_ind = torch.cat([self.topo_knn_ind, topo_ind], 0)
        self.topo_knn_mask = torch.cat([self.topo_knn_mask, topo_mask], 0)

        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)
        if verbose:
            logger.info(f"Topology updated in {time.time() - start_t:.2f}s")
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def update_topology(
        self,
        verbose=False,
        curve_mask=None,
        multilevel_update_flag=True,
    ):
        # * do some other checks here
        if len(self.node_grouping.unique()) != len(self.unique_grouping):
            print()
        assert len(self.node_grouping.unique()) == len(self.unique_grouping), (
            f"{len(self.node_grouping.unique())} vs {len(self.unique_grouping)}"
        )

        # get eval-train state
        assert self.training, "Topology update must be in training mode"
        start_t = time.time()
        if self.fixed_topology_flag:
            logger.warning(
                "Topology update is disabled due to the fixed_topology_flag, but multi-level arap topo is still updated without recompute D"
            )
            self.update_multilevel_arap_topo(verbose=verbose)
            return

        unique_group_id = torch.unique(self.node_grouping)
        if len(unique_group_id) == 1 or not self.break_topo_between_group:
            self.D_topo = _compute_curve_topo_dist_(
                curve_xyz=self.node_xyz,
                curve_mask=curve_mask,
                top_k=int(self.topo_dist_top_k.item()),
                max_subsample_T=int(self.topo_sample_T.item()),
            )
            torch.cuda.empty_cache()
        else:
            assert len(unique_group_id) > 1
            self.D_topo = torch.ones(self.M, self.M).to(self.device) * 1e6
            for gid in tqdm(unique_group_id):
                mask = self.node_grouping == gid
                mask2d = mask[:, None] & mask[None]
                _D = _compute_curve_topo_dist_(
                    curve_xyz=self.node_xyz[:, mask],
                    curve_mask=curve_mask[:, mask] if curve_mask is not None else None,
                    top_k=int(self.topo_dist_top_k.item()),
                    max_subsample_T=int(self.topo_sample_T.item()),
                )
                self.D_topo[mask2d] = _D.reshape(-1)

        topo_dist, topo_ind = __compute_topo_ind_from_dist__(
            self.D_topo, self.skinning_k - 1
        )

        self_ind = torch.arange(self.M).to(self.device)
        topo_ind = torch.cat([self_ind[:, None], topo_ind], dim=1)
        topo_th = self.topo_th_ratio * self.spatial_unit
        topo_mask = topo_dist < topo_th
        topo_mask = torch.cat([torch.ones_like(topo_mask[:, :1]), topo_mask], 1)
        self.topo_knn_ind = topo_ind
        self.topo_knn_mask = topo_mask

        if multilevel_update_flag:
            self.update_multilevel_arap_topo(verbose=verbose)
        if verbose:
            logger.info(f"Topology updated in {time.time() - start_t:.2f}s")
        torch.cuda.empty_cache()

    def update_multilevel_arap_topo(self, verbose=False):
        if not self.mlevel_arap_flag:
            return
        topo_th = self.topo_th_ratio * self.spatial_unit
        self.multilevel_arap_edge_list, self.multilevel_arap_dist_list = (
            __compute_multilevel_topo_ind_from_dist__(
                self.D_topo,
                K_list=self.mlevel_k_list,
                subsample_units=[
                    self.spatial_unit * level for level in self.mlevel_list
                ],
                verbose=verbose,
            )
        )
        multilevel_arap_topo_w = []
        for level, w in zip(self.mlevel_list, self.multilevel_arap_dist_list):
            multilevel_arap_topo_w.append(w < topo_th * level)
            if verbose:
                logger.info(
                    f"MultiRes l={level} {multilevel_arap_topo_w[-1].float().mean() * 100.0:.2f}% valid edges"
                )
        self.multilevel_arap_topo_w = multilevel_arap_topo_w

    def get_async_knns(self, t: torch.Tensor, knn_ind: torch.Tensor):
        assert t.ndim == 1 and knn_ind.ndim == 2 and len(t) == len(knn_ind)
        # self.node_XXXX[t,knn_ind]
        with torch.no_grad():
            flat_sk_ind = t[:, None] * self.M + knn_ind
        sk_ref_node_xyz = self.node_xyz.reshape(-1, 3)[flat_sk_ind, :]  # N,K,3
        sk_ref_node_quat = self.node_rotation.reshape(-1, 4)[flat_sk_ind, :]  # N,K,4
        return sk_ref_node_xyz, sk_ref_node_quat

    def get_async_knns_buffer(
        self, buffer: torch.Tensor, t: torch.Tensor, knn_ind: torch.Tensor
    ):
        assert buffer.ndim == 3  # T,M,C
        assert t.ndim == 1 and knn_ind.ndim == 2 and len(t) == len(knn_ind)
        # self.node_XXXX[t,knn_ind]
        with torch.no_grad():
            flat_sk_ind = t[:, None] * self.M + knn_ind
        C = buffer.shape[2]
        ret = buffer.reshape(-1, C)[flat_sk_ind, :]  # N,K,C
        return ret

    def get_skinning_weights(
        self,
        query_xyz: torch.Tensor,
        query_t: torch.Tensor,
        attach_ind: torch.Tensor,
        skinning_weight_correction: torch.Tensor | None = None,
    ):
        assert query_xyz.ndim == 2
        sk_ind = self.topo_knn_ind[attach_ind]
        sk_mask = self.topo_knn_mask[attach_ind]

        if isinstance(query_t, int) or query_t.ndim == 0:
            sk_ref_node_xyz = self.node_xyz[query_t][sk_ind]
            sk_ref_node_quat = self.node_rotation[query_t][sk_ind]
        else:
            sk_ref_node_xyz, sk_ref_node_quat = self.get_async_knns(query_t, sk_ind)

        sq_dist_to_sk_node = (query_xyz[:, None, :] - sk_ref_node_xyz) ** 2
        sq_dist_to_sk_node = sq_dist_to_sk_node.sum(-1)  # N,K
        sk_w_un = (
            torch.exp(
                -sq_dist_to_sk_node / (2 * (self.node_sigma.squeeze(-1) ** 2)[sk_ind])
            )
            + 1e-6
        )  # N,K

        sk_w_un = sk_w_un * sk_mask.float()
        if skinning_weight_correction is not None:
            assert len(skinning_weight_correction) == len(query_xyz)
            assert skinning_weight_correction.shape[1] == self.skinning_k
            if self.w_corr_maintain_sum_flag:
                tmp_sk_w_sum = abs(sk_w_un).sum(-1)[:, None]  # N,1
                sk_w_un = abs(sk_w_un + skinning_weight_correction)
                new_sk_w_sum = torch.clamp(abs(sk_w_un).sum(-1), min=1e-6)[
                    :, None
                ]  # N,1
                factor = tmp_sk_w_sum / new_sk_w_sum
                sk_w_un = sk_w_un * factor
            else:
                sk_w_un = abs(sk_w_un + skinning_weight_correction)
        sk_w_sum = sk_w_un.sum(-1)
        sk_w = sk_w_un / torch.clamp(sk_w_sum, min=1e-6)[:, None]
        return sk_ind, sk_w, sk_w_sum, sk_ref_node_xyz, sk_ref_node_quat

    ###############################################################
    # * Node Control
    ###############################################################

    @torch.no_grad()
    def resample_node(self, resample_factor=1.0, use_mask=False, resample_ind=None):
        old_M = self.M
        if resample_ind is None:
            resample_ind = resample_curve(
                D=_compute_curve_topo_dist_(
                    self.node_xyz,
                    curve_mask=self.node_certain if use_mask else None,
                    top_k=int(self.topo_dist_top_k.item()),
                    max_subsample_T=int(self.topo_sample_T.item()),
                ),
                sample_margin=resample_factor * self.spatial_unit,
                mask=self.node_certain,
                verbose=True,
            )
        if len(resample_ind) < self.min_node_num:
            logger.warning(
                f"Resample node num {len(resample_ind)} is less than {self.min_node_num}, skip node resample"
            )
            self.update_topology()
            return torch.arange(self.M).to(resample_ind)
        new_node_xyz = self.node_xyz[:, resample_ind]
        new_node_quat = self.node_rotation[:, resample_ind]
        new_node_sigma_logit = self.node_sigma_logit[resample_ind]
        with torch.no_grad():
            self.node_xyz = nn.Parameter(new_node_xyz)
            self.node_rotation = nn.Parameter(new_node_quat)
            self.node_sigma_logit = nn.Parameter(new_node_sigma_logit)
            self.node_certain = self.node_certain[:, resample_ind]
            self.node_grouping = self.node_grouping[resample_ind]
            if len(self.node_grouping.unique()) < len(self.unique_grouping):
                logger.warning(
                    f"Resample node totally removed some groups {self.node_grouping.unique()} <- {self.unique_grouping}"
                )
                self.unique_grouping = torch.unique(self.node_grouping)

        self.update_topology()
        logger.info(
            f"Resample node from {old_M} to {self.M}, with curve_mask={use_mask}"
        )
        self.summary()
        return resample_ind

    @torch.no_grad()
    def append_nodes_pnt(
        self,
        optimizer,
        new_node_xyz,
        new_node_quat,
        new_tid,
        chunk_size=512,
    ):
        if self.M + len(new_node_xyz) > self.max_node_num:
            logger.warning(
                f"Node num is more than {self.max_node_num}, skip node append"
            )
            return False

        start_t = time.time()

        device = new_node_xyz.device
        total_new_nodes = len(new_node_xyz)
        processed_nodes = 0

        while len(new_node_xyz) > 0:
            # Process nodes in chunks
            chunk_size_actual = min(chunk_size, len(new_node_xyz))

            # Get existing nodes at new_tid
            existing_node_xyz = self.node_xyz[new_tid]
            x_batch = None
            y_batch_chunk = None

            # Batch all nodes at once for distance calculation
            if len(new_node_xyz) > chunk_size_actual:
                # Calculate distances to all existing nodes
                x_batch = torch.zeros(
                    existing_node_xyz.shape[0], dtype=torch.long, device=device
                )
                y_batch = torch.zeros(
                    new_node_xyz.shape[0], dtype=torch.long, device=device
                )

                _, all_col = knn(
                    existing_node_xyz,
                    new_node_xyz,
                    k=1,
                    batch_x=x_batch,
                    batch_y=y_batch,
                )

                # Calculate distances for all nodes
                nearest_xyz_all = existing_node_xyz[all_col]
                dist_sq_all = torch.sum((new_node_xyz - nearest_xyz_all) ** 2, dim=1)

                # Select chunk_size nodes with smallest distances
                nearest_ind = torch.argsort(dist_sq_all)[:chunk_size_actual]
                nearest_mask = torch.zeros(
                    len(new_node_xyz), dtype=torch.bool, device=device
                )
                nearest_mask[nearest_ind] = True
            else:
                # Process all remaining nodes
                nearest_mask = torch.ones(
                    len(new_node_xyz), dtype=torch.bool, device=device
                )
                nearest_ind = torch.arange(len(new_node_xyz), device=device)

            # Extract working nodes
            working_node_xyz = new_node_xyz[nearest_mask]
            working_node_quat = new_node_quat[nearest_mask]

            # Get group IDs from nearest existing nodes
            if len(new_node_xyz) > chunk_size_actual:
                # Get nearest existing nodes for selected chunk
                x_batch_chunk = torch.zeros(
                    existing_node_xyz.shape[0], dtype=torch.long, device=device
                )
                y_batch_chunk = torch.zeros(
                    working_node_xyz.shape[0], dtype=torch.long, device=device
                )

                _, col_chunk = knn(
                    existing_node_xyz,
                    working_node_xyz,
                    k=1,
                    batch_x=x_batch_chunk,
                    batch_y=y_batch_chunk,
                )
                new_group_id = self.node_grouping[col_chunk]
            else:
                # For the final chunk, use pre-computed indices
                assert x_batch is not None and y_batch_chunk is not None
                _, final_col = knn(
                    existing_node_xyz,
                    working_node_xyz,
                    k=1,
                    batch_x=x_batch,
                    batch_y=y_batch_chunk,
                )
                new_group_id = self.node_grouping[final_col]

            # Warp nodes to all time frames
            node_xyz_all_times = []
            node_quat_all_times = []

            for _t in tqdm(range(self.T), desc="Warping to time frames"):
                # Find nearest nodes at time _t
                node_pos_t = self.node_xyz[_t]
                x_batch_t = torch.zeros(
                    node_pos_t.shape[0], dtype=torch.long, device=device
                )
                y_batch_t = torch.zeros(
                    working_node_xyz.shape[0], dtype=torch.long, device=device
                )

                _, nearest_node_ind = knn(
                    node_pos_t,
                    working_node_xyz,
                    k=1,
                    batch_x=x_batch_t,
                    batch_y=y_batch_t,
                )

                # Warp nodes
                working_node_xyz_t, working_node_fr_t = self.warp(
                    attach_node_ind=nearest_node_ind,
                    query_xyz=working_node_xyz,
                    query_dir=q2R(working_node_quat),
                    query_tid=new_tid
                    * torch.ones_like(nearest_node_ind, device=device),
                    target_tid=_t,
                    dyn_o_flag=True,
                )

                node_xyz_all_times.append(working_node_xyz_t)

                if working_node_fr_t is not None:
                    node_quat_all_times.append(matrix_to_quaternion(working_node_fr_t))
                else:
                    # Use identity quaternions if no rotation returned
                    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                    node_quat_all_times.append(
                        identity_quat.repeat(len(working_node_xyz), 1)
                    )

            # Append nodes to scaffold
            to_append_node_xyz = torch.stack(node_xyz_all_times, 0)
            to_append_node_quat = torch.stack(node_quat_all_times, 0)

            self.append_nodes_traj(
                optimizer, to_append_node_xyz, to_append_node_quat, new_group_id
            )

            # Update remaining nodes
            new_node_xyz = new_node_xyz[~nearest_mask]
            new_node_quat = new_node_quat[~nearest_mask]

            # Update topology
            self.incremental_topology()

            processed_nodes += chunk_size_actual
            logger.info(f"Processed {processed_nodes}/{total_new_nodes} nodes")

        logger.info(f"Grow nodes in {time.time() - start_t:.2f}s")
        return True

    @torch.no_grad()
    def append_nodes_traj(
        self,
        optimizer,
        new_node_xyz_traj,
        new_node_q_traj,
        new_group_id,
        new_node_sigma_logit=None,
    ):
        # grow a node when know the full trajectory
        N = new_node_q_traj.shape[1]
        if new_node_sigma_logit is None:
            new_node_sigma_logit = self.sig_invact(
                torch.ones(N, device=self.device) * self.init_sigma
            )
            new_node_sigma_logit = new_node_sigma_logit[:, None].expand(
                -1, self.node_sigma_logit.shape[1]
            )
        d = {
            "node_xyz": new_node_xyz_traj,
            "node_rotation": new_node_q_traj,
            "node_sigma": new_node_sigma_logit,
        }
        spec = {"node_xyz": 1, "node_rotation": 1}
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d, spec)
        self.node_xyz = optimizable_tensors["node_xyz"]
        self.node_rotation = optimizable_tensors["node_rotation"]
        self.node_sigma_logit = optimizable_tensors["node_sigma"]

        # ! always append invalid
        append_valid_mask = torch.zeros_like(new_node_xyz_traj[..., 0]).bool()
        self.node_certain = torch.cat([self.node_certain, append_valid_mask], 1)  # T,M'
        self.node_grouping = torch.cat([self.node_grouping, new_group_id], 0)
        return

    @torch.no_grad()
    def remove_nodes(self, optimizer, node_prune_mask):
        if self.M <= self.min_node_num:
            logger.warning(
                f"Node num is less than {self.min_node_num}, skip node prune"
            )
            return
        # node_prune_mask: if remove -> True
        spec = ["node_xyz", "node_rotation", "node_sigma"]
        optimizable_tensors = prune_optimizer(
            optimizer,
            ~node_prune_mask,
            specific_names=spec,
        )
        self.node_xyz = optimizable_tensors["node_xyz"]
        self.node_rotation = optimizable_tensors["node_rotation"]
        self.node_sigma_logit = optimizable_tensors["node_sigma"]
        logger.info(f"Node prune: [-{(node_prune_mask).sum()}]; now has {self.M} nodes")
        self.node_certain = self.node_certain[:, ~node_prune_mask]
        self.node_grouping = self.node_grouping[~node_prune_mask]
        self.decremental_topology(node_prune_mask, verbose=True)
        return

    ###############################################################
    # * reg
    ###############################################################

    def get_optimizable_list(
        self,
        lr_np=0.0001,
        lr_nq=0.0001,
        lr_nsig=0.00001,
    ):
        ret = []
        if lr_np is not None:
            ret.append({"params": [self.node_xyz], "lr": lr_np, "name": "node_xyz"})
        if lr_nq is not None:
            ret.append(
                {"params": [self.node_rotation], "lr": lr_nq, "name": "node_rotation"}
            )
        if lr_nsig is not None:
            ret.append(
                {
                    "params": [self.node_sigma_logit],
                    "lr": lr_nsig,
                    "name": "node_sigma",
                }
            )
        return ret

    @torch.no_grad()
    def mask_xyz_grad(self, mask):
        # for init stage maintain the observed xyz unchanged
        assert mask.shape == self.node_xyz.shape[:2]
        assert self.node_xyz.grad is not None
        mask = mask.to(self.node_xyz)
        self.node_xyz.grad = self.node_xyz.grad * mask[..., None]
        return

    ##################################################
    # warp helper
    ##################################################
    @torch.no_grad()
    def identify_nearest_node_id(self, query_xyz, query_tid, query_group_id=None):
        # Handle scalar query_tid
        if isinstance(query_tid, int) or query_tid.ndim == 0:
            query_tid = torch.ones_like(query_xyz[:, 0]).int() * query_tid

        N = len(query_tid)
        assert query_tid.ndim == 1
        assert len(query_xyz) == N and query_xyz.shape == (N, 3)

        ret_id = -torch.ones_like(query_tid, dtype=torch.long)

        # Get unique time frames
        unique_times = torch.unique(query_tid)

        for t in unique_times:
            mask = query_tid == t
            if not mask.any():
                continue

            query_pos_t = query_xyz[mask]

            if query_group_id is not None:
                group_id_t = query_group_id[mask]
                unique_groups = torch.unique(group_id_t)

                # Precompute node positions for this time
                node_pos_all = self.node_xyz[t]
                node_ids_all = torch.arange(self.M, device=self.device)

                nearest_node_ind_t = -torch.ones(
                    mask.sum(), dtype=torch.long, device=self.device
                )

                for gid in unique_groups:
                    mask_gid = group_id_t == gid
                    if not mask_gid.any():
                        continue

                    # Get node mask for this group
                    node_mask = self.node_grouping == gid
                    if not node_mask.any():
                        # Fallback to all nodes
                        node_mask_use = torch.ones_like(
                            self.node_grouping, dtype=torch.bool
                        )
                    else:
                        node_mask_use = node_mask

                    # Get node positions and IDs for this group
                    node_pos = node_pos_all[node_mask_use]
                    node_ids = node_ids_all[node_mask_use]

                    # Skip if no nodes
                    if node_pos.shape[0] == 0:
                        continue

                    # Find nearest neighbors
                    query_pos_g = query_pos_t[mask_gid]

                    # Use batch indices for knn
                    x_batch = torch.zeros(
                        node_pos.shape[0], dtype=torch.long, device=self.device
                    )
                    y_batch = torch.zeros(
                        query_pos_g.shape[0], dtype=torch.long, device=self.device
                    )

                    _, col = knn(
                        node_pos, query_pos_g, k=1, batch_x=x_batch, batch_y=y_batch
                    )
                    nearest_node_ind_t[mask_gid] = node_ids[col]

                ret_id[mask] = nearest_node_ind_t

            else:
                # Non-grouping mode
                node_pos = self.node_xyz[t]

                # Use batch indices
                x_batch = torch.zeros(
                    node_pos.shape[0], dtype=torch.long, device=self.device
                )
                y_batch = torch.zeros(
                    query_pos_t.shape[0], dtype=torch.long, device=self.device
                )

                _, col = knn(
                    node_pos, query_pos_t, k=1, batch_x=x_batch, batch_y=y_batch
                )
                ret_id[mask] = col.int()

            # Final check
            if not (ret_id >= 0).all():
                missing = (ret_id < 0).sum().item()
                raise ValueError(
                    f"{missing} queries failed to find nearest node. IDs: {ret_id[ret_id < 0]}"
                )

            return ret_id

        @torch.no_grad()
        def __compute_R_from_xyz__(self):
            init_node_R = torch.eye(3).to(self.device)[None].repeat(self.M, 1, 1)
            new_R_list = [init_node_R]
            nn_ind, nn_mask = self.topo_knn_ind, self.topo_knn_mask
            for new_tid in tqdm(range(1, self.T)):
                # ! do to the first frame!

                src_xyz, dst_xyz = self.node_xyz[new_tid - 1], self.node_xyz[new_tid]

                # DEBUG: seems the old version does not normalize the centroid, may affect??
                # DEBUG: the old version has bug, seems the w is not used anymore ...

                src_p = src_xyz[nn_ind] - src_xyz[:, None]
                dst_q = dst_xyz[nn_ind] - dst_xyz[:, None]
                w = nn_mask
                w = w / w.sum(dim=-1, keepdim=True)  # M,K

                src_centroid = (src_p * w[..., None]).sum(1)
                dst_centroid = (dst_q * w[..., None]).sum(1)
                src_p = src_p - src_centroid[:, None]
                dst_q = dst_q - dst_centroid[:, None]

                W = torch.einsum("nki,nkj->nkij", src_p, dst_q)

                W = W * w[:, :, None, None]
                W = W.sum(1)

                U, s, V = torch.svd(W.double())
                # ! warning, torch's svd has W = U @ torch.diag(s) @ (V.T)
                # U, s, V = U.float(), s.float(), V.float()
                # R_star = V @ (U.T) # ! handling flipping
                R_tmp = torch.einsum("nij,nkj->nik", V, U)
                det = torch.det(R_tmp)
                dia = torch.ones(len(det), 3).to(det)
                dia[:, -1] = det
                Sigma = torch.diag_embed(dia)
                V = torch.einsum("nij,njk->nik", V, Sigma)
                R_star = torch.einsum("nij,nkj->nik", V, U)
                # dst = R_star @ src
                next_R = torch.einsum(
                    "nij,njk->nik", R_star.double(), new_R_list[-1].double()
                )
                new_R_list.append(next_R.float())
            new_R_list = torch.stack(new_R_list, dim=0)
            new_q_list = matrix_to_quaternion(new_R_list)
            return new_q_list

        ##################################################
        # Time densify
        ##################################################

        @torch.no_grad()
        def resample_time(self, new_tids, new_node_certain, mode="linear"):
            assert new_tids.max() <= self.t_list.max(), "no extrapolate"
            assert new_tids.min() >= self.t_list.min(), "no extrapolate"
            new_xyz_list, new_quat_list = [], []
            for new_t in tqdm(new_tids):
                left_t = self.t_list[self.t_list <= new_t].max()
                left_ind = (self.t_list == left_t).float().argmax()
                assert left_ind >= 0 and left_ind < self.T
                l_xyz, l_quat = self.node_xyz[left_ind], self.node_rotation[left_ind]
                if left_t == new_t:
                    new_xyz_list.append(l_xyz)
                    new_quat_list.append(l_quat)
                    continue
                right_t = self.t_list[self.t_list >= new_t].min()
                assert left_t < new_t < right_t
                right_ind = (self.t_list == right_t).float().argmax()
                assert right_ind >= 0 and right_ind < self.T
                r_xyz, r_quat = self.node_xyz[right_ind], self.node_rotation[right_ind]
                if right_t == new_t:
                    new_xyz_list.append(r_xyz)
                    new_quat_list.append(r_quat)
                    continue
                # print(left_t, right_t)
                # print(left_ind, right_ind)
                l_w = float(right_t - new_t) / float(right_t - left_t)
                r_w = float(new_t - left_t) / float(right_t - left_t)

                if mode == "dq":
                    l_dq = Rt2dq(q2R(l_quat), l_xyz)
                    r_dq = Rt2dq(q2R(r_quat), r_xyz)
                    dq = l_dq * l_w + r_dq * r_w
                    dq = dq2unitdq(dq)
                    new_R, new_xyz = dq2Rt(dq)
                    new_quat = matrix_to_quaternion(new_R)
                elif mode == "linear":
                    new_xyz = l_xyz * l_w + r_xyz * r_w
                    new_quat = l_quat * l_w + r_quat * r_w
                else:
                    raise NotImplementedError()

                new_xyz_list.append(new_xyz)
                new_quat_list.append(new_quat)

            new_xyz_list = torch.stack(new_xyz_list, 0)
            new_quat_list = torch.stack(new_quat_list, 0)

            self.node_xyz.data = new_xyz_list
            self.node_rotation.data = new_quat_list
            assert new_node_certain.shape[1] == self.M and len(new_tids) == len(
                new_node_certain
            )
            self.node_certain.data = new_node_certain

            # update t_list
            remap = [torch.where(new_tids == t)[0][0].item() for t in self.t_list.cpu()]
            self.t_list = new_tids.to(self.t_list)
            return remap  # new index in the new list


##################################################################################
##################################################################################
def robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k: int):
    # src, dst: T,N,C
    # src_m, dst_m: T,N bool
    _d = (src - dst).norm(dim=-1)
    T = len(src)
    _m = src_m.bool() & dst_m.bool()  # T, chunk
    _d = _d * _m
    _m_cnt = _m.sum(dim=0)
    # shrink the top-k ind if the visible count is less than T
    top_k_select_id = torch.ceil(top_k * _m_cnt / T).long() - 1  # - 1 is for 0-index
    top_k_select_id = torch.clamp(top_k_select_id, 0, top_k - 1)
    top_k_value, _ = torch.topk(_d, k=top_k, dim=0, largest=True)  # K,Chunk
    # * because top_k always < T, so the selected top_k value should always be valid!
    assert top_k_select_id.max() < top_k
    d = torch.gather(top_k_value, 0, top_k_select_id[None]).squeeze(0)  # Chunk
    return d


@torch.no_grad()
def __query_distance_to_curve__(
    q_curve_xyz,
    b_curve_xyz,
    q_mask=None,
    b_mask=None,
    max_subsample_T=80,
    top_k=8,
    chunk=65536 // 8,
):
    # q_curve_xyz: T,N,3; b_curve_xyz: T,M,3, usually N << M
    # return: N,M

    T, N, _ = q_curve_xyz.shape
    M = b_curve_xyz.shape[1]
    assert q_curve_xyz.shape[0] == b_curve_xyz.shape[0]

    if q_mask is None:
        q_mask = torch.ones(T, N).bool().to(q_curve_xyz.device)
    else:
        q_mask = q_mask.bool().to(q_curve_xyz.device)
    if b_mask is None:
        b_mask = torch.ones(T, M).bool().to(b_curve_xyz.device)
    else:
        b_mask = b_mask.bool().to(b_curve_xyz.device)

    t_choice = None
    if T > max_subsample_T:
        t_choice = torch.randperm(T)[:max_subsample_T]
        t_choice = torch.sort(t_choice)[0]

    q_curve_xyz = q_curve_xyz[t_choice] if T > max_subsample_T else q_curve_xyz
    q_mask = q_mask[t_choice] if T > max_subsample_T else q_mask
    b_curve_xyz = b_curve_xyz[t_choice] if T > max_subsample_T else b_curve_xyz
    b_mask = b_mask[t_choice] if T > max_subsample_T else b_mask
    T = len(q_curve_xyz)

    cur = 0
    ret = torch.zeros(0).to(q_curve_xyz)

    # not triangle, but all pairs
    all_pair_ind = torch.meshgrid(torch.arange(N), torch.arange(M))
    all_pair_ind = torch.stack(all_pair_ind, -1)  # N,M,2, the first is the N index
    all_pair_ind = all_pair_ind.reshape(-1, 2)

    while cur < len(all_pair_ind):
        ind = all_pair_ind[cur : cur + chunk]
        src = q_curve_xyz[:, ind[:, 0]]  # T,Chunk,3
        dst = b_curve_xyz[:, ind[:, 1]]
        src_m = q_mask[:, ind[:, 0]]  # T, chunk
        dst_m = b_mask[:, ind[:, 1]]  # T, chunk
        d = robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k)
        ret = torch.cat([ret, d])
        cur = cur + chunk
    ret = ret.reshape(N, M)
    return ret


@torch.no_grad()
def _compute_curve_topo_dist_(
    curve_xyz: torch.Tensor,
    curve_mask: torch.Tensor | None = None,
    max_subsample_T=80,
    top_k=8,
    chunk=65536,
) -> torch.Tensor:
    # * this  function have to handle size N~10k-30k, T<max_subsample_T
    # curve_xyz: T,N,3

    T, N = curve_xyz.shape[:2]
    if curve_mask is None:
        curve_mask = torch.ones(T, N).bool().to(curve_xyz.device)
    else:
        curve_mask = curve_mask.bool().to(curve_xyz.device)

    t_choice = None
    if T > max_subsample_T:
        t_choice = torch.randperm(T)[:max_subsample_T]
        t_choice = torch.sort(t_choice)[0]

    xyz = curve_xyz[t_choice] if T > max_subsample_T else curve_xyz
    mask = curve_mask[t_choice] if T > max_subsample_T else curve_mask

    T = len(xyz)

    # prepare the ind pair to compute the dist (half pairs because of symmetric)
    ind_pair = torch.triu_indices(N, N).permute(1, 0)  # N,2
    P = len(ind_pair)
    cur = 0
    flat_D = torch.zeros(0).to(xyz)
    while cur < P:
        ind = ind_pair[cur : cur + chunk]

        src = xyz[:, ind[:, 0]]  # T,Chunk,3
        dst = xyz[:, ind[:, 1]]  # T,Chunk,3
        src_m = mask[:, ind[:, 0]]  # T, chunk
        dst_m = mask[:, ind[:, 1]]  # T, chunk

        d = robust_curve_dist_kernel(src, dst, src_m, dst_m, top_k)

        flat_D = torch.cat([flat_D, d])
        cur = cur + chunk
    # convert the list to upper and lower triangle
    _d = torch.zeros(N, N).to(xyz)
    print(f"N:{N}")
    print(f"_d shape:{_d.shape} type:{_d.dtype}")
    print(f"flat_D shape:{flat_D.shape} type:{flat_D.dtype}")
    print(f"curve_xyz shape:{curve_xyz.shape} type:{curve_xyz.dtype}")
    print(f"ind_pair shape:{ind_pair.shape} type:{ind_pair.dtype}")
    # assert False, _d.shape
    # _d[ind_pair[:, 0], ind_pair[:, 1]] = flat_D
    # _d[ind_pair[:, 1], ind_pair[:, 0]] = flat_D
    _d.index_put_((ind_pair[:, 0], ind_pair[:, 1]), flat_D)
    _d.index_put_((ind_pair[:, 1], ind_pair[:, 0]), flat_D)

    # make distance that is zero to be Huge
    # ! set distance to self as inf as well, outside this will handle skinning to self
    invalid_mask = _d == 0
    _d[invalid_mask] = 1e10
    return _d


##################################################################################
##################################################################################


@torch.no_grad()
def __compute_topo_ind_from_dist__(dist, K):
    knn_dist, knn_ind = torch.topk(dist, K, dim=-1, largest=False)
    return knn_dist, knn_ind


@torch.no_grad()
def __compute_multilevel_topo_ind_from_dist__(
    dist: torch.Tensor,
    K_list: list,
    subsample_units: list,
    shrink_level=False,
    verbose=False,
):
    torch.cuda.empty_cache()
    assert not shrink_level, (
        "Warning, shrink the level will makes the multi-level arap not helping!, shoudl always do dense!"
    )
    # dist: N,N, K_list and subsample_unit list are list of knn and subsample units
    N, _ = dist.shape
    current_set = torch.arange(N, device=dist.device)

    edge_list, dist_list = [], []
    for k, unit in zip(K_list, subsample_units):
        torch.cuda.empty_cache()
        if not shrink_level:
            # everytime the source coordinate is the original one
            current_set = torch.arange(N, device=dist.device)
        # subsample the dist curve by the units
        assert current_set.max() < N and current_set.min() >= 0
        cur_D = dist[current_set][:, current_set]
        resample_ind = resample_curve(cur_D, unit, mask=None, verbose=verbose)
        if len(resample_ind) < 1:
            logger.info("No resampled nodes, early stop!")
            break
        assert resample_ind.max() < len(current_set) and resample_ind.min() >= 0
        sub_D = cur_D[
            :, resample_ind
        ]  # ! before convert to global ind, first subsample the sub_D
        resample_ind = current_set[resample_ind].clone()  # ! in original N len

        # for all previous set curve, find the K nearest subset curve
        nearest_ind = torch.topk(
            sub_D, min(k, len(resample_ind)), dim=1, largest=False
        ).indices
        nearest_ind = resample_ind[nearest_ind]  # ! in original N len
        src_ind = current_set[:, None].expand(-1, min(k, len(resample_ind)))
        edge = torch.stack([src_ind, nearest_ind], dim=-1).reshape(-1, 2)  # N,k,2
        assert edge.max() < N and edge.min() >= 0
        dist_list.append(dist[edge[..., 0], edge[..., 1]])  # N,k
        edge_list.append(edge)

        # save the subset as current set
        current_set = resample_ind
        if verbose:
            logger.info(
                f"level {len(edge_list)} with margin={unit:.4f} k={k} |Set|={len(current_set)} |E|={len(dist_list[-1])}"
            )
    torch.cuda.empty_cache()
    return edge_list, dist_list


@torch.no_grad()
def resample_curve(D, sample_margin, mask=None, verbose=False):
    N = D.shape[0]
    assert D.shape == (N, N)
    if mask is None:
        rank_inds = torch.randperm(N)
    else:
        rank_inds = torch.argsort(mask.sum(0), descending=True)
    sampled_inds = rank_inds[:1]
    # for ind in tqdm(rank_inds[1:]):
    for ind in rank_inds[1:]:
        if D[ind][sampled_inds].min() > sample_margin:
            sampled_inds = torch.cat([sampled_inds, ind[None]])
    # sort sampled_inds
    sampled_inds = torch.sort(sampled_inds).values
    if verbose:
        logger.info(
            f"SCF Resample with th={sample_margin:.4f} N={len(sampled_inds)} out of {N} ({len(sampled_inds) / N * 100.0:.2f}%)"
        )
    assert sampled_inds.max() < N and sampled_inds.min() >= 0
    return sampled_inds


# @torch.no_grad()
# def __identify_spatial_unit_from_curves__(xyz, mask, K=10):
#     T = len(xyz)
#     dist_list = []
#     for t in tqdm(range(T)):
#         _p = xyz[t][mask[t]]
#         dist_sq, _, _ = knn_points(_p[None], _p[None], K=min(K, mask[t].sum()))
#         if dist_sq.shape[1] == 0:
#             continue
#         dist_list.append(dist_sq.squeeze(0).median())
#     dist_list = torch.tensor(dist_list)
#     dist_list = torch.clamp(dist_list, 1e-6, 1e6)
#     unit = torch.sqrt(dist_list.median())
#     assert not torch.isnan(unit), f"{dist_list}"
#     return float(unit)


@torch.no_grad()
def __identify_spatial_unit_from_curves__(xyz: torch.Tensor, mask: torch.Tensor, K=10):
    T = len(xyz)
    dist_list = []
    for t in tqdm(range(T)):
        _p = xyz[t][mask[t]]
        if _p.shape[0] < 2:
            continue
        # Create KNN graph
        edge_index = knn_graph(_p, k=min(K, _p.shape[0] - 1), loop=False)
        # Calculate distances for edges
        row, col = edge_index
        dist_sq = torch.sum((_p[row] - _p[col]) ** 2, dim=1)
        if dist_sq.numel() == 0:
            continue
        # Get median distance
        dist_list.append(dist_sq.median())
    if not dist_list:
        return 1.0
    dist_list = torch.tensor(dist_list)
    dist_list = torch.clamp(dist_list, 1e-6, 1e6)
    unit = torch.sqrt(dist_list.median())
    assert not torch.isnan(unit), f"{dist_list}"
    return float(unit)
