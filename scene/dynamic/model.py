from typing import Any, Dict
import torch_geometric.nn.pool as pyg_pool
import torch
from torch import nn
from loguru import logger
from matplotlib import pyplot as plt
from utils3d.torch import (
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points

from lib_mosca.mosca import DQ_EPS, MoSca
from scene.cfg import GaussianSplattingConfig
from viz_utils import q2R


# TODO: flexiblly swtich between GS mu parameterization, for later efficiency, have to directly save the location.


class DynamicScene(nn.Module):
    #################################################################
    # * Trace buffers and parameters
    #################################################################
    # * buffers
    # max_scale
    # min_sacle
    # max_sph_order

    # attach_ind
    # ref_time

    # xyz_gradient_accum
    # xyz_gradient_denom
    # max_radii2D

    # * parameters
    # _xyz
    # _rotation
    # _scaling
    # _opacity
    # _features_dc
    # _features_rest
    # _skinning_weight
    # _dynamic_logit
    #################################################################

    # ! note: use_dyn_o = False so permanently remove

    def __init__(
        self,
        scf: MoSca,
        means: torch.Tensor,  # N,3
        rgbs: torch.Tensor,  # N,3
        scales: torch.Tensor,  # N,3
        times: torch.Tensor,  # N, 1
        cfg: GaussianSplattingConfig,
        quats: torch.Tensor | None = None,  # N,4
        opacities: torch.Tensor | None = None,  # N
    ) -> None:
        N = rgbs.shape[0]
        self.cfg = cfg
        self.scf = scf

        # gsplat expects: [N, (sph_order + 1)², 3]
        num_coeffs = (cfg.sph_order + 1) ** 2
        shs = torch.zeros(N, num_coeffs, 3, device=self.device)
        shs[:, 0, :] = rgbs
        if cfg.sph_order > 0:
            std = 0.01
            shs[:, 1:, :] = torch.randn(N, num_coeffs - 1, 3, device=self.device) * std

        quats = torch.ones((means.shape[0], 4))

        attach_inds = torch.zeros(N)
        for tid in torch.unique(times):
            mask = times == tid
            _, attach_ind, _ = knn_points(
                means[None], self.scf._node_xyz[tid][None], K=1
            )
            attach_node_xyz = self.scf._node_xyz[tid][attach_ind[0, :, 0]]
            attach_node_R_wi = q2R(self.scf._node_rotation[tid][attach_ind[0, :, 0]])
            means[mask] = torch.einsum(
                "nji,nj->ni", attach_node_R_wi, means[mask] - attach_node_xyz
            )
            R_new = torch.einsum("nji,njk->nik", attach_node_R_wi, q2R([mask]))
            quats[mask] = matrix_to_quaternion(R_new)
            attach_inds[mask] = attach_ind

        self.params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(
                    torch.ones(means.shape[0]) if opacities is None else opacities
                ),
                "quats": nn.Parameter(quats),
                "shs": nn.Parameter(shs),
                "attach_inds": nn.Parameter(attach_inds, requires_grad=False),  # N
            }
        )

        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            name: cfg.get_optimizer()(
                params=param,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            for name, param in self.params.items()
            if param.requires_grad
        }
        self.strategy = cfg.strategy
        self.state = self.strategy.initialize_state()
        self.summary()

        self.skinning_weight = nn.Parameter(torch.zeros(self.N, self.scf.skinning_k))
        self._dynamic_logit = nn.Parameter(self.o_inv_act(torch.ones(self.N, 1) * 0.99))
        self.register_buffer("ref_time", torch.zeros(self.N).long())  # N
        # * init states
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        # * for tracing the correspondence gradient
        self.register_buffer("corr_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("corr_gradient_denom", torch.zeros(self.N).long())
        self.summary()

    @classmethod
    def load_from_ckpt(cls, ckpt, device=torch.device("cuda:0")):
        # first recover the
        scf_sub_ckpt = {k[4:]: v for k, v in ckpt.items() if k.startswith("scf.")}
        scf = MoSca.load_from_ckpt(scf_sub_ckpt, device=device)
        if "scf.mlevle_detach_nn_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.mlevel_detach_nn_flag"] = torch.tensor(True)
        if "scf.mlevel_detach_self_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.mlevel_detach_self_flag"] = torch.tensor(False)
        if "scf.w_corr_maintain_sum_flag" not in ckpt.keys():
            # old ckpt
            ckpt["scf.w_corr_maintain_sum_flag"] = torch.tensor(False)
        model = cls(
            scf=scf,
            init_N=ckpt["_xyz"].shape[0],
            max_sph_order=ckpt["max_sph_order"],
        )
        if "corr_gradient_accum" not in ckpt.keys():
            logger.info("Old ckpt, add corr_gradient_accum")
            ckpt["corr_gradient_accum"] = ckpt["xyz_gradient_accum"].clone() * 0.0
            ckpt["corr_gradient_denom"] = ckpt["xyz_gradient_denom"].clone() * 0
        if "max_node_num" not in ckpt.keys():
            logger.info("Old ckpt, add max_node_num")
            ckpt["max_node_num"] = torch.tensor([100000])
        model.load_state_dict(ckpt, strict=True)
        model.summary()
        return model

    def summary(self):
        logger.info(
            f"DenseDynGaussian: {self.N / 1000.0:.1f}K points; {self.M} Nodes; K={self.scf.skinning_k if self.scf is not None else None}; and {self.T} time step"
        )

    @property
    def get_group(self):
        # fetch the group id from the attached to nearest node
        group_id = self.scf._node_grouping[self.attach_ind]
        return group_id

    def get_xyz(self):
        nn_ref_node_xyz, sk_ref_node_quat = self.scf.get_async_knns(
            self.ref_time, self.attach_ind[:, None]
        )
        nn_ref_node_xyz = nn_ref_node_xyz.squeeze(1)
        nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
        return (
            torch.einsum("nij,nj->ni", nn_ref_node_R_wi, self.means) + nn_ref_node_xyz
        )

    def get_R_mtx(self):
        if self.leaf_local_flag:
            _, sk_ref_node_quat = self.scf.get_async_knns(
                self.ref_time, self.attach_ind[:, None]
            )
            nn_ref_node_R_wi = q2R(sk_ref_node_quat.squeeze(1))
            return torch.einsum("nij,njk->nik", nn_ref_node_R_wi, q2R(self.quats))
        else:
            return q2R(self.quats)

    def set_surface_deform(self):
        # * different to using RBF field approximate the deformation field (more flexible when changing the scf topology), set the deformation to surface model, the skinning is saved on each GS
        logger.info("ED Model convert to surface mode")
        self.w_correction_flag = torch.tensor(True).to(self.device)
        self.scf.fixed_topology_flag = torch.tensor(True).to(self.device)

    def eval(self):
        # bake the buffer and enable fast inference FPS
        baked_query_xyz = self.get_xyz()
        baked_query_dir = self.get_R_mtx()
        self.register_buffer("baked_query_xyz", baked_query_xyz)
        self.register_buffer("baked_query_dir", baked_query_dir)
        (
            baked_sk_ind,
            baked_sk_w,
            sk_w_sum,
            baked_sk_ref_node_xyz,
            baked_sk_ref_node_quat,
        ) = self.scf.get_skinning_weights(
            query_xyz=baked_query_xyz,
            query_t=self.ref_time,
            attach_ind=self.attach_ind,
            skinning_weight_correction=(
                self.skinning_weight if self.w_correction_flag else None
            ),
        )
        baked_dyn_o = torch.ones_like(sk_w_sum) * (1.0 - DQ_EPS)
        self.register_buffer("baked_sk_ind", baked_sk_ind)
        self.register_buffer("baked_sk_w", baked_sk_w)
        self.register_buffer("baked_sk_ref_node_xyz", baked_sk_ref_node_xyz)
        self.register_buffer("baked_sk_ref_node_quat", baked_sk_ref_node_quat)
        self.register_buffer("baked_dyn_o", baked_dyn_o)
        self.fast_inference_flag = True
        logger.warning(("ED Model convert to inference mode"))
        return super().eval()

    def forward(
        self, t: int, nn_fusion=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert t < self.T, "t is out of range!"

        if not self.training:
            mu_live, fr_live = self.scf.fast_warp(
                target_tid=t,
                # all below are baked
                sk_ind=self.baked_sk_ind,
                sk_w=self.baked_sk_w,
                sk_ref_node_xyz=self.baked_sk_ref_node_xyz,
                sk_ref_node_quat=self.baked_sk_ref_node_quat,
                dyn_o=self.baked_dyn_o,
                query_xyz=self.baked_query_xyz,
                query_dir=self.baked_query_dir,
            )
        else:
            mu_live, fr_live = self.scf.warp(
                attach_node_ind=self.attach_ind,
                query_xyz=self.get_xyz(),
                query_dir=self.get_R_mtx(),
                query_tid=self.ref_time,
                target_tid=t,
                skinning_w_corr=(
                    self.skinning_weight if self.w_correction_flag else None
                ),
            )

        return (
            mu_live,
            fr_live,
            self.params["scales"],
            self.params["opacities"],
            self.params["shs"],
        )

    ######################################################################
    # * Node Control
    ######################################################################

    @torch.no_grad()
    def prune_nodes(self, optimizer, prune_sk_th=0.02, viz_fn=None):
        # if a node is not carrying leaves, and the effect to all neighbors are small. then can prune it; and also update the knn skinning weight, during this update, also have to be careful about the inner scf-knn-ind for scf, only replace some where!!!

        acc_w = self.get_node_sinning_w_acc("max")
        # viz
        if viz_fn is not None:
            plt.figure(figsize=(10, 5))
            plt.hist(acc_w.cpu().numpy(), bins=100)
            plt.plot([prune_sk_th, prune_sk_th], [0, 100], "r--")
            plt.title("Node Max supporting sk-w hist")
            plt.savefig(f"{viz_fn}prune_sk_hist.jpg")
            plt.close()

        prune_mask_sk = acc_w < prune_sk_th  # if prune, true

        # also check whether this node carries some leaves
        supporting_node_id = torch.unique(self.attach_ind)
        prune_mask_carry = torch.ones(self.M, device=self.device).bool()
        prune_mask_carry[supporting_node_id] = False

        node_prune_mask = prune_mask_sk & prune_mask_carry
        logger.info(
            f"Prune {node_prune_mask.sum()} nodes (max_sk<th={prune_sk_th}) with carrying check ({node_prune_mask.float().mean() * 100.0:.3f}%)"
        )

        prune_M = node_prune_mask.sum().item()
        if prune_M == 0:
            return

        # first remove the leaves
        # ! actually this is not used in our case for now
        leaf_pruning_mask = (
            self.attach_ind[:, None]
            == torch.arange(self.M, device=self.device)[None, node_prune_mask]
        ).any(-1)
        if leaf_pruning_mask.any():
            self._prune_points(optimizer, leaf_pruning_mask)

        if self.w_correction_flag:
            # identify the sk corr that related to the old node
            sk_corr_affect_mask = node_prune_mask[
                self.scf.topo_knn_ind[self.attach_ind]
            ]
            logger.warning(
                f"Prune under surface mode, check {sk_corr_affect_mask.sum()}({sk_corr_affect_mask.float().mean() * 100:.3f}%) sk_corr to be updated"
            )
            # ! later make these position sk to be zero

        # then update the attach ind
        new_M = self.M - prune_M
        ind_convert = torch.ones(self.M, device=self.device).long() * -1
        ind_convert[~node_prune_mask] = torch.arange(new_M, device=self.device)
        self.attach_ind = ind_convert[self.attach_ind]

        # finally remove the nodes
        self.scf.remove_nodes(optimizer, node_prune_mask)

        # now update the sk corr again, make sure the updated = 0.0
        if self.w_correction_flag:
            _, sk_w, sk_w_sum, _, _ = self.scf.get_skinning_weights(
                query_xyz=self.get_xyz(),
                query_t=self.ref_time,
                attach_ind=self.attach_ind,
                # skinning_weight_correction=self._skinning_weight,
            )
            sk_w_field = sk_w * sk_w_sum[:, None]
            new_sk_corr = self.skinning_weight.clone()
            new_sk_corr[sk_corr_affect_mask] = -sk_w_field[sk_corr_affect_mask]
            # replace sk_w again

            optimizable_tensors = replace_tensor_to_optimizer(
                optimizer,
                [new_sk_corr],
                ["skinning_w"],
            )
            self.skinning_weight = optimizable_tensors["skinning_w"]

    ########
    # * Another densification
    ########
    @torch.no_grad()
    def gradient_based_node_densification(
        self,
        optimizer,
        gradient_th,
        resample_factor=1.0,
        max_gs_per_new_node=100000,  # 32
    ):
        logger.info("Starting Grandient Based Node Densification...")
        grad = self.corr_gradient_accum / (self.corr_gradient_denom + 1e-6)
        candidate_mask = grad > gradient_th  # N

        if not candidate_mask.any():
            logger.info("No node to densify")
            return

        gs_mu_list, gs_fr_list = [], []
        for t in range(self.T):
            mu, fr, _, _, _ = self.forward(t)
            gs_mu_list.append(mu[candidate_mask])
            assert fr is not None
            gs_fr_list.append(fr[candidate_mask])

        gs_mu_list = torch.stack(gs_mu_list, dim=0)  # T,N,3
        gs_fr_list = torch.stack(gs_fr_list, dim=0)  # T,N,3,3

        # subsample the gs_mu_list

        resample_ind = resample_curve(
            D=_compute_curve_topo_dist_(
                gs_mu_list,
                curve_mask=None,
                top_k=self.scf.topo_dist_top_k,
                max_subsample_T=self.scf.topo_sample_T,
            ),
            sample_margin=resample_factor * self.scf.spatial_unit,
            mask=None,
            verbose=True,
        )

        if self.scf.M + len(resample_ind) > self.max_node_num:
            logger.warning("Node num exceeds the maximum limit, do not increase")
            return

        new_node_xyz = gs_mu_list[:, resample_ind]
        new_node_quat = matrix_to_quaternion(gs_fr_list[:, resample_ind])

        # append these new node into scf
        old_M = self.scf.M
        self.scf.append_nodes_traj(
            optimizer,
            new_node_xyz,
            new_node_quat,
            torch.zeros(new_node_xyz.shape[1]).to(self.scf._node_grouping),
        )
        self.scf.incremental_topology()  # ! manually must set this

        # ! warning, for now, all appended nodes are set to have group-id=0

        # find these gs's original attached node
        original_attach_ind = self.attach_ind[candidate_mask][resample_ind]

        new_attach_ind_list, new_gs_ind_list = [], []
        for _i in range(new_node_xyz.shape[1]):
            _attach_ind = old_M + _i
            # the same carrying leaves duplicate them
            neighbors_mask = self.attach_ind == original_attach_ind[_i]
            if not neighbors_mask.any():
                continue
            # ! bound the number of leaves here
            # !  WARNING, THIS MODIFICATION IS AFTER MANY BASE VERSION, BE CAREFUL
            if neighbors_mask.long().sum() > float(max_gs_per_new_node):
                # random sample max_gs_per_new_node and mark the flag
                neighbors_ind = torch.arange(self.N, device=self.device)[neighbors_mask]
                neighbors_ind = neighbors_ind[
                    torch.randperm(neighbors_ind.shape[0])[:max_gs_per_new_node]
                ]
                neighbors_mask = torch.zeros_like(neighbors_mask)
                neighbors_mask[neighbors_ind] = True
            #
            gs_ind = torch.arange(self.N, device=self.device)[neighbors_mask]
            new_attach_ind_list.append(torch.ones_like(gs_ind) * _attach_ind)
            new_gs_ind_list.append(gs_ind)
        if len(new_attach_ind_list) == 0:
            logger.info("No new leaves to append")
            return
        new_attach_ind = torch.cat(new_attach_ind_list, dim=0)
        new_gs_ind = torch.cat(new_gs_ind_list, dim=0)

        logger.info(
            f"Append {new_node_xyz.shape[1]} new nodes and dup {new_gs_ind.shape[0]} leaves"
        )

        assert new_gs_ind.max() < self.N, f"{new_gs_ind.max()}, {self.N}"

        self._densification_postprocess(
            optimizer,
            new_xyz=self.means[new_gs_ind].detach().clone(),
            new_r=self.quats[new_gs_ind].detach().clone(),
            new_s_logit=self.scales[new_gs_ind].detach().clone(),
            new_o_logit=self.opacities[new_gs_ind].detach().clone(),
            new_sph_dc=self._features_dc[new_gs_ind].detach().clone(),
            new_sph_rest=self._features_rest[new_gs_ind].detach().clone(),
            new_skinning_w=self.skinning_weight[new_gs_ind].detach().clone(),
            new_dyn_logit=self.o_inv_act(
                torch.ones_like(self.means[new_gs_ind][:, :1]) * 0.99
            ),
        )

        self.attach_ind = torch.cat(
            [self.attach_ind, new_attach_ind.to(self.attach_ind)], dim=0
        )
        assert self.attach_ind.max() < self.scf.M, (
            f"{self.attach_ind.max()}, {self.scf.M}"
        )
        self.ref_time = torch.cat(
            [self.ref_time, self.ref_time.clone()[new_gs_ind]], dim=0
        )
        assert self.ref_time.max() < self.T, f"{self.ref_time.max()}, {self.T}"

        self.clean_corr_control_record()

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_pre_backward(
            self.params, self.optimizers, self.state, step, info
        )

    def step_post_backward(self, step: int, info: Dict[str, Any]):
        self.strategy.step_post_backward(
            self.params, self.optimizers, self.state, step, info
        )  # type: ignore


def subsample_vtx(vtx, voxel_size):
    # vtx: N,3
    # according to surfelwarp global_config.h line28 d_node_radius=0.025 meter; and warpfieldinitializer.cpp line39 the subsample voxel is 0.7 * 0.025 meter
    # reference: VoxelSubsamplerSorting.cu line  119
    # Here use use the mean of each voxel cell
    pooling_ind = pyg_pool.voxel_grid(pos=vtx, size=voxel_size)
    unique_ind, compact_ind = pooling_ind.unique(return_inverse=True)
    candidate = torch.scatter_reduce(
        input=torch.zeros(len(unique_ind), 3).to(vtx),
        src=vtx,
        index=compact_ind[:, None].expand_as(vtx),
        dim=0,
        reduce="mean",
        # dim_size=len(unique_ind),
        include_self=False,
    )
    assert not (candidate == 0).all(dim=-1).any(), "voxel resampling has an error!"
    return candidate
