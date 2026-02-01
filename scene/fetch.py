# pyright: reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false
import logging
import torch


@torch.no_grad()
def fetch_leaves_in_world_frame(
    cams: MonocularCameras,
    n_attach: int,  # if negative use all
    #
    input_mask_list,
    input_dep_list,
    input_rgb_list,
    input_normal_list=None,  # ! in new version, do not use this
    input_inst_list=None,  # ! in new version, do not use this
    #
    save_xyz_fn=None,
    start_t=0,
    end_t=-1,
    t_list=None,
    subsample=1,
    squeeze_normal_ratio=1.0,
):
    device = cams.rel_focal.device

    if end_t == -1:
        end_t = cams.T
    if t_list is None:
        t_list = list(range(start_t, end_t))
    if subsample > 1:
        logging.info(f"2D Subsample {subsample} for fetching ...")

    mu_list, quat_list, scale_list, rgb_list, time_index_list = [], [], [], [], []
    inst_list = []  # collect the leaf id as well

    for t in tqdm(t_list):
        mask2d = input_mask_list[t].bool()
        H, W = mask2d.shape
        if subsample > 1:
            mask2d[::subsample, ::subsample] = False

        dep_map = input_dep_list[t].clone()
        cam_pcl = cams.backproject(
            cams.get_homo_coordinate_map(H, W)[mask2d].clone(), dep_map[mask2d]
        )
        cam_R_wc, cam_t_wc = cams.Rt_wc(t)
        mu = cams.trans_pts_to_world(t, cam_pcl)
        rgb_map = input_rgb_list[t].clone()
        rgb = rgb_map[mask2d]
        K = cams.K(H, W)
        radius = cam_pcl[:, -1] / (0.5 * K[0, 0] + 0.5 * K[1, 1]) * float(subsample)
        scale = torch.stack([radius / squeeze_normal_ratio, radius, radius], dim=-1)
        time_index = torch.ones_like(mu[:, 0]).long() * t

        if input_normal_list is not None:
            nrm_map = input_normal_list[t].clone()
            cam_nrm = nrm_map[mask2d]
            nrm = F.normalize(torch.einsum("ij,nj->ni", cam_R_wc, cam_nrm), dim=-1)
            rx = nrm.clone()
            ry = F.normalize(torch.cross(rx, mu, dim=-1), dim=-1)
            rz = F.normalize(torch.cross(rx, ry, dim=-1), dim=-1)
            rot = torch.stack([rx, ry, rz], dim=-1)
        else:
            rot = torch.eye(3)[None].expand(len(radius), -1, -1)
        quat = matrix_to_quaternion(rot)

        mu_list.append(mu.cpu())
        quat_list.append(quat.cpu())
        scale_list.append(scale.cpu())
        rgb_list.append(rgb.cpu())
        time_index_list.append(time_index.cpu())

        if input_inst_list is not None:
            inst_map = inst_list[t].clone()
            inst = inst_map[mask2d]
            inst_list.append(inst.cpu())

    mu_all = torch.cat(mu_list, 0)
    quat_all = torch.cat(quat_list, 0)
    scale_all = torch.cat(scale_list, 0)
    rgb_all = torch.cat(rgb_list, 0)

    logging.info(
        f"Fetching {n_attach / 1000.0:.3f}K out of {len(mu_all) / 1e6:.3}M pts"
    )
    if n_attach > len(mu_all) or n_attach <= 0:
        choice = torch.arange(len(mu_all))
    else:
        choice = torch.randperm(len(mu_all))[:n_attach]

    # make gs5 param (mu, fr, s, o, sph) no rescaling
    mu_init = mu_all[choice].clone()
    q_init = quat_all[choice].clone()
    s_init = scale_all[choice].clone()
    o_init = torch.ones(len(choice), 1).to(mu_init)
    rgb_init = rgb_all[choice].clone()
    time_init = torch.cat(time_index_list, 0)[choice]
    if len(inst_list) > 0:
        inst_all = torch.cat(inst_list, 0)
        inst_init = inst_all[choice].clone().to(device)
    else:
        inst_init = None
    if save_xyz_fn is not None:
        np.savetxt(
            save_xyz_fn,
            torch.cat([mu_init, rgb_init * 255], 1).detach().cpu().numpy(),
            fmt="%.6f",
        )
    torch.cuda.empty_cache()
    return (
        mu_init.to(device),
        q_init.to(device),
        s_init.to(device),
        o_init.to(device),
        rgb_init.to(device),
        inst_init,
        time_init.to(device),
    )
