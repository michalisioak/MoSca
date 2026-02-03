from dataclasses import dataclass
from loguru import logger
import torch
from tqdm import tqdm
from lib_moca.camera import MonocularCameras
from lib_prior.prior_loading import Saved2D

from scene.static import StaticScene
from scene.dynamic import DynamicScene

@dataclass
class PhotometricFitConfig:
    optim_cam_after_steps=0
    total_steps=8000
    topo_update_feq=50
    skinning_corr_start_steps=1e10
    s_gs_ctrl_start_ratio=0.2
    s_gs_ctrl_end_ratio=0.9
    d_gs_ctrl_start_ratio=0.2
    d_gs_ctrl_end_ratio=0.9


def static_photometric_fit(
        s2d: Saved2D,
        cams: MonocularCameras,
        s_model: StaticScene,
        d_model: DynamicScene,
        cfg: PhotometricFitConfig,
        
        # # optim
        # optimizer_cfg: OptimCFG = OptimCFG(
        #     lr_cam_f=0.0,
        #     lr_cam_q=0.0001,
        #     lr_cam_t=0.0001,
        #     lr_p=0.0003,
        #     lr_q=0.002,
        #     lr_s=0.01,
        #     lr_o=0.1,
        #     lr_sph=0.005,
        #     # dyn
        #     lr_np=0.001,
        #     lr_nq=0.01,
        #     lr_w=0.3,
        # ),
        # cfg loss
        lambda_rgb=1.0,
        lambda_dep=1.0,
        lambda_mask=0.5,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100.0,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        lambda_small_w_reg=0.0,
        #
        lambda_track=0.0,
        track_flow_chance=0.0,
        track_flow_interval_candidates=[1],
        track_loss_clamp=100.0,
        track_loss_protect_steps=100,
        track_loss_interval=3,  # 1/3 steps are used for track loss and does not count the grad
        track_loss_start_step=-1,
        track_loss_end_step=100000,
        ######
        reg_radius=None,
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        phase_name="photometric",
        use_decay=True,
        decay_start=2000,
        temporal_diff_shift=[1, 3, 6],
        temporal_diff_weight=[0.6, 0.3, 0.1],
        # * error grow
        dyn_error_grow_steps=[],
        dyn_error_grow_th=0.2,
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # ! warning, the corr loss is in pixel int unit!!
        dyn_node_densify_steps=[],
        dyn_node_densify_grad_th=0.2,
        dyn_node_densify_record_start_steps=2000,
        dyn_node_densify_max_gs_per_new_node=100000,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        # other
        random_bg=False,
        default_bg_color=[1.0, 1.0, 1.0],
        # * DynGS cleaning
        photo_s2d_trans_steps=[],
    ):
        logger.info("Starting Photometric Fit...")
        torch.cuda.empty_cache()
        optimizer_dynamic = None
        dst_xyz_cam = None
        reg_tids = None

        
        n_frame = 1

        d_flag = d_model is not None

        corr_flag = lambda_track > 0.0 and d_flag
        if corr_flag:
            logging.info(
                f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
            )

        cam_param_list = cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
        if len(cam_param_list) > 0:
            optimizer_cam = torch.optim.Adam(
                cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
            )
        else:
            optimizer_cam = None
        if use_decay:
            gs_scheduling_func_dict, cam_scheduling_func_dict = (
                optimizer_cfg.get_scheduler(total_steps=total_steps - decay_start)
            )
        else:
            gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}

        loss_rgb_list, loss_dep_list, loss_nrm_list = [], [], []
        loss_mask_list = []
        loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

        loss_arap_coord_list, loss_arap_len_list = [], []
        loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
        loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
        s_n_count_list, d_n_count_list = [], []
        d_m_count_list = []
        loss_sds_list = []
        loss_small_w_list = []
        loss_track_list = []

        s_gs_ctrl_start = int(cfg.total_steps * cfg.s_gs_ctrl_start_ratio)
        d_gs_ctrl_start = int(cfg.total_steps * cfg.d_gs_ctrl_start_ratio)
        s_gs_ctrl_end = int(cfg.total_steps * cfg.s_gs_ctrl_end_ratio)
        d_gs_ctrl_end = int(cfg.total_steps * cfg.d_gs_ctrl_end_ratio)
        assert s_gs_ctrl_start >= 0
        assert d_gs_ctrl_start >= 0

        latest_track_event = 0
        base_u, base_v = np.meshgrid(np.arange(s2d.W), np.arange(s2d.H))
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=s2d.rgb.device).long()

        # prepare a color-plate for seamntic rendering
        if d_flag:
            # ! for now the group rendering only works for dynamic joitn mode
            n_group_static = len(s_model.group_id.unique())
            assert d_model.scf != None, "d_model is None"
            n_group_dynamic = len(d_model.scf.unique_grouping)
            color_plate = get_colorplate(n_group_static + n_group_dynamic)
            # random permute
            color_permute = torch.randperm(len(color_plate))
            color_plate = color_plate[color_permute]
            s_model.get_cate_color(
                color_plate=color_plate[:n_group_static].to(s_model.device)
            )
            d_model.get_cate_color(
                color_plate=color_plate[n_group_static:].to(d_model.device)
            )

        for step in tqdm(range(cfg.total_steps)):
            # * control the w correction
            if d_flag and step == skinning_corr_start_steps:
                logging.info(
                    f"at {step} stop all the topology update and add skinning weight correction"
                )
                d_model.set_surface_deform()
            corr_exe_flag = (
                corr_flag
                and step > latest_track_event + track_loss_protect_steps
                and step % track_loss_interval == 0
                and step >= track_loss_start_step
                and step < track_loss_end_step
            )
            if optimizer_cam is not None:
                optimizer_cam.zero_grad()
            cams.zero_grad()
            s2d.zero_grad()
       
            if step % topo_update_feq == 0:
                assert d_model.scf != None
                d_model.scf.update_topology()

            if step > decay_start:
                for k, v in gs_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, optimizer_static)
                    if d_flag:
                        update_learning_rate(v(step), k, optimizer_dynamic)
                if optimizer_cam is not None:
                    for k, v in cam_scheduling_func_dict.items():
                        update_learning_rate(v(step), k, optimizer_cam)

            view_ind_list = np.random.choice(cams.T, n_frame, replace=False).tolist()
            if corr_exe_flag:
                # select another ind different than the view_ind_list
                corr_dst_ind_list, corr_flow_flag_list = [], []
                for view_ind in view_ind_list:
                    flow_flag = np.random.rand() < track_flow_chance
                    corr_flow_flag_list.append(flow_flag)
                    if flow_flag:
                        corr_dst_ind_candidates = []
                        for flow_interval in track_flow_interval_candidates:
                            if view_ind + flow_interval < cams.T:
                                corr_dst_ind_candidates.append(view_ind + flow_interval)
                            if view_ind - flow_interval >= 0:
                                corr_dst_ind_candidates.append(view_ind - flow_interval)
                        corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                        corr_dst_ind_list.append(corr_dst_ind)
                    else:
                        corr_dst_ind = view_ind
                        while corr_dst_ind == view_ind:
                            corr_dst_ind = np.random.choice(cams.T)
                        corr_dst_ind_list.append(corr_dst_ind)
                corr_dst_ind_list = np.array(corr_dst_ind_list)
            else:
                corr_dst_ind_list = view_ind_list
                corr_flow_flag_list = [False] * n_frame

            render_dict_list, corr_render_dict_list = [], []
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0
            loss_mask = 0.0
            loss_track = 0.0
            for _inner_loop_i, view_ind in enumerate(view_ind_list):
                dst_ind = corr_dst_ind_list[_inner_loop_i]
                flow_flag = corr_flow_flag_list[_inner_loop_i]
                gs5 = [list(s_model())]

                add_buffer = None
                if corr_exe_flag:
                    # ! detach bg pts
                    assert d_model != None, "d_model is None"
                    dst_xyz = torch.cat([gs5[0][0].detach(), d_model(dst_ind)[0]], 0)
                    dst_xyz_cam = cams.trans_pts_to_cam(dst_ind, dst_xyz)
                    if GS_BACKEND in ["native_add3"]:
                        add_buffer = dst_xyz_cam

                if d_flag:
                    gs5.append(list(d_model(view_ind)))
                if random_bg:
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = default_bg_color  # [1.0, 1.0, 1.0]
                if GS_BACKEND in ["native_add3"]: # WARNING TYPO
                    # the render internally has another protection, because if not set, the grad has bug
                    bg_color += [0.0, 0.0, 0.0]

                render_dict = render(
                    gs5,
                    s2d.H,
                    s2d.W,
                    cams.K(s2d.H, s2d.W),
                    cams.T_cw(view_ind),
                    bg_color=bg_color,
                    add_buffer=add_buffer,
                )
                render_dict_list.append(render_dict)

                # compute losses
                rgb_sup_mask = s2d.get_mask_by_key(sup_mask_type)[view_ind]
                _l_rgb, _, _, _ = compute_rgb_loss(
                    s2d.rgb[view_ind].detach().clone(), render_dict, rgb_sup_mask
                )
                dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind]
                _l_dep, _, _, _ = compute_dep_loss(
                    s2d.dep[view_ind].detach().clone(),
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant,
                )
                loss_rgb = loss_rgb + _l_rgb
                loss_dep = loss_dep + _l_dep

                if corr_exe_flag:
                    # * Track Loss
                    if GS_BACKEND in ["native_add3"]:
                        corr_render_dict = render_dict
                        rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                    else:
                        corr_render_dict = render(
                            # # ! use detached bg gs
                            # [[it.detach() for it in gs5[0]], gs5[1]],
                            # ! debug, align wiht .bck old version
                            gs5,
                            s2d.H,
                            s2d.W,
                            cams.K(s2d.H, s2d.W),
                            cams.T_cw(view_ind),
                            bg_color=[0.0, 0.0, 0.0],
                            colors_precomp=dst_xyz_cam,
                        )
                        rendered_xyz_map = corr_render_dict["rgb"].permute(
                            1, 2, 0
                        )  # H,W,3
                    corr_render_dict_list.append(corr_render_dict)
                    # get the flow
                    with torch.no_grad():
                        if flow_flag:
                            flow_ind = s2d.flow_ij_to_listind_dict[(view_ind, dst_ind)]
                            flow = s2d.flow[flow_ind].detach().clone()
                            flow_mask = s2d.flow_mask[flow_ind].detach().clone().bool()
                            track_src = base_uv.clone().detach()[flow_mask]
                            flow = flow[flow_mask]
                            track_dst = track_src.float() + flow
                        else:
                            # contruct target by track
                            track_valid = (
                                s2d.track_mask[view_ind] & s2d.track_mask[dst_ind]
                            )
                            track_src = s2d.track[view_ind][track_valid][..., :2]
                            track_dst = s2d.track[dst_ind][track_valid][..., :2]
                        src_fetch_index = (
                            track_src[:, 1].long() * s2d.W + track_src[:, 0].long()
                        )
                    if len(track_src) == 0:
                        _loss_track = torch.zeros_like(_l_rgb)
                    else:
                        warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[
                            src_fetch_index
                        ]
                        # filter the pred, only add loss to points that are infront of the camera
                        track_loss_mask = warped_xyz_cam[:, 2] > 1e-4
                        if track_loss_mask.sum() == 0:
                            _loss_track = torch.zeros_like(_l_rgb)
                        else:
                            pred_track_dst = cams.project(warped_xyz_cam)
                            L = min(s2d.W, s2d.H)
                            pred_track_dst[:, :1] = (
                                (pred_track_dst[:, :1] + s2d.W / L) / 2.0 * L
                            )
                            pred_track_dst[:, 1:] = (
                                (pred_track_dst[:, 1:] + s2d.H / L) / 2.0 * L
                            )
                            _loss_track = (pred_track_dst - track_dst).norm(dim=-1)[
                                track_loss_mask
                            ]
                            _loss_track = torch.clamp(
                                _loss_track, 0.0, track_loss_clamp
                            )
                            _loss_track = _loss_track.mean()
                else:
                    _loss_track = torch.zeros_like(_l_rgb)
                loss_track = loss_track + _loss_track

                

                ############
                if d_flag and lambda_mask > 0.0:
                    # * do the mask loss, including the background
                    s_cate_sph, s_gid2color = s_model.get_cate_color(
                        perm=torch.randperm(len(s_model.group_id.unique()))
                    )
                    assert d_model.scf != None
                    d_cate_sph, d_gid2color = d_model.get_cate_color(
                        perm=torch.randperm(len(d_model.scf.unique_grouping))
                    )
                    with torch.no_grad():
                        inst_map = s2d.inst[view_ind]
                        gt_mask = torch.zeros_like(s2d.rgb[0])
                        for gid, color in d_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                        for gid, color in s_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                    gs5[1][-1] = d_cate_sph
                    gs5[0][-1] = s_cate_sph
                    render_dict = render(
                        gs5,
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                    )
                    pred_mask = render_dict["rgb"].permute(1, 2, 0)
                    l_mask = torch.nn.functional.mse_loss(pred_mask, gt_mask)
                    loss_mask = loss_mask + l_mask
                    # imageio.imsave(f"./debug/mask.jpg", pred_mask.detach().cpu())
                    # imageio.imsave(f"./debug/gt_mask.jpg", gt_mask.detach().cpu())
                else:
                    loss_mask = torch.zeros_like(loss_rgb)

            if d_flag:
                _l = max(0, view_ind_list[0] - reg_radius)
                _r = min(cams.T, view_ind_list[0] + 1 + reg_radius)
                reg_tids = torch.arange(_l, _r, device=s_model.device)
            if (lambda_arap_coord > 0.0 or lambda_arap_len > 0.0) and d_flag:
                assert d_model.scf != None
                loss_arap_coord, loss_arap_len = d_model.scf.compute_arap_loss(
                    reg_tids,
                    temporal_diff_shift=temporal_diff_shift,
                    temporal_diff_weight=temporal_diff_weight,
                )
                assert torch.isnan(loss_arap_coord).sum() == 0
                assert torch.isnan(loss_arap_len).sum() == 0
            else:
                loss_arap_coord = torch.zeros_like(torch.tensor(loss_rgb))
                loss_arap_len = torch.zeros_like(torch.tensor(loss_rgb))

            if (
                lambda_vel_xyz_reg > 0.0
                or lambda_vel_rot_reg > 0.0
                or lambda_acc_xyz_reg > 0.0
                or lambda_acc_rot_reg > 0.0
            ) and d_flag:
                assert d_model.scf != None
                (
                    loss_vel_xyz_reg,
                    loss_vel_rot_reg,
                    loss_acc_xyz_reg,
                    loss_acc_rot_reg,
                ) = d_model.scf.compute_vel_acc_loss(reg_tids)
            else:
                loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                    loss_acc_rot_reg
                ) = torch.zeros_like(torch.tensor(loss_rgb))

            if d_flag:
                loss_small_w = abs(d_model._skinning_weight).mean()
            else:
                loss_small_w = torch.zeros_like(torch.tensor(loss_rgb))

            loss = (
                loss_rgb * lambda_rgb
                + loss_dep * lambda_dep
                + loss_mask * lambda_mask
                + loss_nrm * lambda_normal
                + loss_dep_nrm_reg * lambda_depth_normal
                + loss_distortion_reg * lambda_distortion
                + loss_arap_coord * lambda_arap_coord
                + loss_arap_len * lambda_arap_len
                + loss_vel_xyz_reg * lambda_vel_xyz_reg
                + loss_vel_rot_reg * lambda_vel_rot_reg
                + loss_acc_xyz_reg * lambda_acc_xyz_reg
                + loss_acc_rot_reg * lambda_acc_rot_reg
                + loss_small_w * lambda_small_w_reg
                + loss_track * lambda_track
            )

            s_model.step_pre_backward(step)
            d_model.step_pre_backward(step)
            loss.backward()
            s_model.step_post_backward(step)
            d_model.step_post_backward(step)

            if step >= cfg.optim_cam_after_steps and optimizer_cam is not None:
                optimizer_cam.step()

            # d_model to s_model transfer [1] copy the d_gs5
            dynamic_to_static_transfer_flag = step in photo_s2d_trans_steps and d_flag
            if dynamic_to_static_transfer_flag:
                with torch.no_grad():
                    # before the gs control to append full opacity GS
                    random_select_t = np.random.choice(cams.T)
                    assert d_model != None, "d_model is None"
                    trans_d_gs5 = d_model(random_select_t)
                    logging.info(f"Transfer dynamic to static at step={step}")

            

            # d_model to s_model transfer [2] append to static model
            if dynamic_to_static_transfer_flag:
                s_model.append_gs(optimizer_static, *trans_d_gs5, new_group_id=None)

            if d_flag and step in dyn_node_densify_steps:
                d_model.gradient_based_node_densification(
                    optimizer_dynamic,
                    gradient_th=dyn_node_densify_grad_th,
                    max_gs_per_new_node=dyn_node_densify_max_gs_per_new_node,
                )

            # error grow
            if d_flag and step in dyn_error_grow_steps:
                error_grow_dyn_model(
                    s2d,
                    cams,
                    s_model,
                    d_model,
                    optimizer_dynamic,
                    step,
                    dyn_error_grow_th,
                    dyn_error_grow_num_frames,
                    dyn_error_grow_subsample,
                    viz_dir=self.viz_dir,
                    opacity_init_factor=self.opacity_init_factor,
                )
            if d_flag and step in dyn_scf_prune_steps:
                d_model.prune_nodes(
                    optimizer_dynamic,
                    prune_sk_th=dyn_scf_prune_sk_th,
                    viz_fn=osp.join(self.viz_dir, f"scf_node_prune_at_step={step}"),
                )

            loss_rgb_list.append(loss_rgb.item())
            loss_dep_list.append(loss_dep.item())
            loss_nrm_list.append(loss_nrm.item())
            loss_mask_list.append(loss_mask.item())

            loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
            loss_distortion_reg_list.append(loss_distortion_reg.item())

            loss_arap_coord_list.append(loss_arap_coord.item())
            loss_arap_len_list.append(loss_arap_len.item())
            loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
            loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
            loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
            loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
            s_n_count_list.append(s_model.N)
            d_n_count_list.append(d_model.N if d_flag else 0)
            d_m_count_list.append(d_model.M if d_flag else 0)

            loss_small_w_list.append(loss_small_w.item())
            loss_track_list.append(loss_track.item())

            # viz
            viz_flag = viz_interval > 0 and (step % viz_interval == 0)
            if viz_flag:

                if d_flag:
                    viz_hist(d_model, self.viz_dir, f"{phase_name}_step={step}_dynamic")
                    viz_dyn_hist(
                        d_model.scf,
                        self.viz_dir,
                        f"{phase_name}_step={step}_dynamic",
                    )
                    viz_path = osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_3dviz.mp4"
                    )
                    viz3d_total_video(
                        cams,
                        d_model,
                        0,
                        cams.T - 1,
                        save_path=viz_path,
                        res=480,  # 240
                        s_model=s_model,
                    )

                    # * viz grouping
                    if lambda_mask > 0.0:
                        d_model.return_cate_colors_flag = True
                        viz_path = osp.join(
                            self.viz_dir, f"{phase_name}_step={step}_3dviz_group.mp4"
                        )
                        viz3d_total_video(
                            cams,
                            d_model,
                            0,
                            cams.T - 1,
                            save_path=viz_path,
                            res=480,  # 240
                            s_model=s_model,
                        )
                        viz2d_total_video(
                            viz_vid_fn=osp.join(
                                self.viz_dir,
                                f"{phase_name}_step={step}_2dviz_group.mp4",
                            ),
                            s2d=s2d,
                            start_from=0,
                            end_at=cams.T - 1,
                            skip_t=viz_skip_t,
                            cams=cams,
                            s_model=s_model,
                            d_model=d_model,
                            subsample=1,
                            mask_type=sup_mask_type,
                            move_around_angle_deg=viz_move_angle_deg,
                        )
                        d_model.return_cate_colors_flag = False

                viz_hist(s_model, self.viz_dir, f"{phase_name}_step={step}_static")
                viz2d_total_video(
                    viz_vid_fn=osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_2dviz.mp4"
                    ),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    subsample=1,
                    mask_type=sup_mask_type,
                    move_around_angle_deg=viz_move_angle_deg,
                )

            if viz_cheap_interval > 0 and (
                step % viz_cheap_interval == 0 or step == total_steps - 1
            ):
                # viz the accumulated grad
                with torch.no_grad():
                    photo_grad = [
                        s_model.xyz_gradient_accum
                        / torch.clamp(s_model.xyz_gradient_denom, min=1e-6)
                    ]
                    corr_grad = [torch.zeros_like(photo_grad[0])]
                    if d_flag:
                        photo_grad.append(
                            d_model.xyz_gradient_accum
                            / torch.clamp(d_model.xyz_gradient_denom, min=1e-6)
                        )
                        corr_grad.append(
                            d_model.corr_gradient_accum
                            / torch.clamp(d_model.corr_gradient_denom, min=1e-6)
                        )

                    photo_grad = torch.cat(photo_grad, 0)
                    viz_grad_color = (
                        torch.clamp(photo_grad, 0.0, d_gs_ctrl_cfg.densify_max_grad)
                        / d_gs_ctrl_cfg.densify_max_grad
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(photo_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_photo_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                    corr_grad = torch.cat(corr_grad, 0)
                    viz_grad_color = (
                        torch.clamp(corr_grad, 0.0, dyn_node_densify_grad_th)
                        / dyn_node_densify_grad_th
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(corr_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_corr_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                fig = plt.figure(figsize=(30, 8))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_nrm", loss_nrm_list),
                        ("loss_mask", loss_mask_list),
                        ("loss_sds", loss_sds_list),
                        ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                        ("loss_distortion_reg", loss_distortion_reg_list),
                        ("loss_arap_coord", loss_arap_coord_list),
                        ("loss_arap_len", loss_arap_len_list),
                        ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                        ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                        ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                        ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                        ("loss_small_w", loss_small_w_list),
                        ("loss_track", loss_track_list),
                        ("S-N", s_n_count_list),
                        ("D-N", d_n_count_list),
                        ("D-M", d_m_count_list),
                    ]
                ):
                    plt.subplot(2, 10, plt_i + 1)
                    value_end = 0 if len(plt_pack[1]) == 0 else plt_pack[1][-1]
                    plt.plot(plt_pack[1]), plt.title(
                        plt_pack[0] + f" End={value_end:.4f}"
                    )
                plt.savefig(
                    osp.join(self.viz_dir, f"{phase_name}_optim_loss_step={step}.jpg")
                )
                plt.close()

        # save static, camera and dynamic model
        s_save_fn = osp.join(
            self.log_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth"
        )
        torch.save(s_model.state_dict(), s_save_fn)
        torch.save(cams.state_dict(), osp.join(self.log_dir, f"{phase_name}_cam.pth"))

        if d_model is not None:
            d_save_fn = osp.join(
                self.log_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth"
            )
            torch.save(d_model.state_dict(), d_save_fn)

        # viz
        fig = plt.figure(figsize=(30, 8))
        for plt_i, plt_pack in enumerate(
            [
                ("loss_rgb", loss_rgb_list),
                ("loss_dep", loss_dep_list),
                ("loss_nrm", loss_nrm_list),
                ("loss_mask", loss_mask_list),
                ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                ("loss_distortion_reg", loss_distortion_reg_list),
                ("loss_arap_coord", loss_arap_coord_list),
                ("loss_arap_len", loss_arap_len_list),
                ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                ("loss_small_w", loss_small_w_list),
                ("loss_track", loss_track_list),
                ("S-N", s_n_count_list),
                ("D-N", d_n_count_list),
                ("D-M", d_m_count_list),
            ]
        ):
            plt.subplot(2, 10, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
        plt.savefig(osp.join(self.log_dir, f"{phase_name}_optim_loss.jpg"))
        plt.close()
        viz2d_total_video(
            viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz.mp4"),
            s2d=s2d,
            start_from=0,
            end_at=cams.T - 1,
            skip_t=viz_skip_t,
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            move_around_angle_deg=viz_move_angle_deg,
            print_text=False,
        )
        viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz.mp4")
        if d_flag:
            viz3d_total_video(
                cams,
                d_model,
                0,
                cams.T - 1,
                save_path=viz_path,
                res=480,
                s_model=s_model,
            )
            if lambda_mask > 0.0:
                # * viz grouping
                d_model.return_cate_colors_flag = True
                s_model.return_cate_colors_flag = True
                viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz_group.mp4")
                viz3d_total_video(
                    cams,
                    d_model,
                    0,
                    cams.T - 1,
                    save_path=viz_path,
                    res=480,
                    s_model=s_model,
                )
                viz2d_total_video(
                    viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz_group.mp4"),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    move_around_angle_deg=viz_move_angle_deg,
                    print_text=False,
                )
                d_model.return_cate_colors_flag = False
                s_model.return_cate_colors_flag = False
        torch.cuda.empty_cache()
        
