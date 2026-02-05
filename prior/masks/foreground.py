import torch


@torch.no_grad()
def identify_fg_mask_by_nearest_curve(
    s2d: Saved2D, cams: MonocularCameras, viz_fname=None
):
    # get global anchor
    curve_xyz, curve_mask, _, _ = get_dynamic_curves(s2d, cams, return_all_curves=True)
    assert curve_xyz.shape[1] == len(s2d.dynamic_track_mask)

    # only consider the valid case
    static_curve_mean = (
        curve_xyz[:, ~s2d.dynamic_track_mask]
        * curve_mask[:, ~s2d.dynamic_track_mask, None]
    ).sum(0, keepdim=True) / curve_mask[:, ~s2d.dynamic_track_mask, None].sum(
        0, keepdim=True
    ).expand(len(curve_xyz), -1, -1)
    curve_xyz[:, ~s2d.dynamic_track_mask] = static_curve_mean
    np.savetxt(
        osp.join(self.viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
        static_curve_mean.reshape(-1, 3).cpu().numpy(),
        fmt="%.4f",
    )

    with torch.no_grad():
        fg_mask_list = []
        for query_tid in tqdm(range(s2d.T)):
            query_dep = s2d.dep[query_tid].clone()  # H,W
            query_xyz_cam = cams.backproject(cams.get_homo_coordinate_map(), query_dep)
            query_xyz_world = cams.trans_pts_to_world(query_tid, query_xyz_cam)  # H,W,3

            # find the nearest distance and acc sk weight
            # use all the curve at this position to id the fg and bg
            _, knn_id, _ = knn_points(
                query_xyz_world.reshape(1, -1, 3), curve_xyz[query_tid][None], K=1
            )
            knn_id = knn_id[0, :, 0]
            fg_mask = s2d.dynamic_track_mask[knn_id].reshape(s2d.H, s2d.W)
            fg_mask_list.append(fg_mask.cpu())
        fg_mask_list = torch.stack(fg_mask_list, 0)
    if viz_fname is not None:
        viz_rgb = s2d.rgb.clone().cpu()
        viz_fg_mask_list = fg_mask_list * s2d.dep_mask.to(fg_mask_list)
        viz_rgb = viz_rgb * viz_fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
            1 - viz_fg_mask_list.float()[..., None]
        )
        imageio.mimsave(osp.join(self.viz_dir, viz_fname), viz_rgb.numpy())

    fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool()
    s2d.register_2d_identification(
        static_2d_mask=~fg_mask_list, dynamic_2d_mask=fg_mask_list
    )
    return fg_mask_list
