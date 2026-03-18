import sys
import os
import os.path as osp
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms

from lib_prior.tracking.spatracker2.models.vggt4track.models.vggt_moe import VGGT4Track
from .spatracker2.utils.visualizer import Visualizer as SpaVizualizer
import logging
import time
import glob
import imageio
import gc

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer
from tracking_utils import (
    seed_everything,
    convert_img_list_to_cotracker_input,
    get_sampling_mask,
    get_uniform_random_queries,
    load_epi_error,
    load_vos,
    viz_queries,
    viz_coverage,
    tracker_get_query_uv,
)

from spatracker2.models.predictor import Predictor


def get_spatracker2(device):
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    # model.spatrack.track_num = 6
    model.eval()
    model.to(device)
    return model


def make_spatracker_input(rgb_list, dep_list):
    # T,H,W,3; T,H,W
    assert rgb_list.ndim == 4 and rgb_list.shape[-1] == 3
    assert dep_list.ndim == 3
    assert len(rgb_list) == len(dep_list)

    input_video = (
        torch.from_numpy(rgb_list).permute(0, 3, 1, 2).float()[None].cuda()
    )  # 1,T,3,H,W
    input_depth = torch.from_numpy(dep_list).float().cuda()[:, None]  # T,1,H,W

    return input_video, input_depth


@torch.no_grad()
def infer_spa_tracker(
    video: torch.Tensor,
    dep: torch.Tensor,
    queries: torch.Tensor,
    model: Predictor,
    K: torch.Tensor | None,
    visiblility_th=0.9,
    device="cuda",
):
    start_t = time.time()
    _, T, _, H, W = video.shape

    torch.cuda.empty_cache()

    assert video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3, (
        "video should have size: 1,T,3,H,W"
    )
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3, (
        "queries should have size: 1,N,3"
    )
    if K is not None:
        K = torch.as_tensor(K).to(device).clone()
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3, (
                "K should have size: T,3,3"
            )
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        K = K[None]  # 1,T,3,3

    video = video.to(device)
    dep = dep.to(device)
    queries = queries.to(device)

    (
        c2w_traj,
        intrs,
        point_map,
        conf_depth,
        pred_tracks,
        track2d_pred,
        pred_visibility,
        conf_pred,
        video,
    ) = model.forward(
        video=video[0].cpu().numpy(),
        queries=queries[0].cpu().numpy(),
        depth=dep.cpu().numpy(),
        intrs=K,
    )
    pred_tracks = pred_tracks[..., :3]
    print(f"pred_tracks shape: {pred_tracks.shape}")
    print(f"pred_visibility: {pred_visibility.squeeze(-1).shape}")
    # tracks =(torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], pred_tracks[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]) # T,N,3
    # print(f"tracks shape:{tracks.shape}")
    # print(f"pred vis shape: {pred_visibility.squeeze(-1).shape} ")
    # pred_visibility = pred_visibility[0]

    end_t = time.time()
    print(f"SpaT 2 bi-directional time cost: {(end_t - start_t) / 60.0:.3f} min")

    return (
        pred_tracks.cpu(),
        pred_visibility.squeeze(-1).cpu(),
        video,
        track2d_pred.cpu(),
        video.cpu(),
    )


@torch.no_grad()
def spatracker2_process_folder(
    working_dir,
    img_list,
    dep_list,
    sample_mask_list,
    model: Predictor,
    total_n_pts,
    K,
    chunk_size=10000,  # designed for 16GB GPU
    save_name="",
    max_viz_cnt=512,
    support_ratio=0.2,
    use_half_res=True,
):
    viz_dir = osp.join(working_dir, "spatracker2_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )

    # Store original dimensions
    orig_H, orig_W = img_list.shape[1:3]

    # If half resolution is requested, downsample inputs
    if use_half_res:
        target_H, target_W = orig_H // 2, orig_W // 2

        # Downsample images
        img_list_resized = []
        for img in img_list:
            img_resized = cv2.resize(
                img, (target_W, target_H), interpolation=cv2.INTER_LINEAR
            )
            img_list_resized.append(img_resized)
        img_list = np.stack(img_list_resized, 0)

        # Downsample depth
        dep_list_resized = []
        for dep in dep_list:
            dep_resized = cv2.resize(
                dep, (target_W, target_H), interpolation=cv2.INTER_NEAREST
            )
            dep_list_resized.append(dep_resized)
        dep_list = np.stack(dep_list_resized, 0)

        # Downsample sample mask
        sample_mask_list_resized = []
        for mask in sample_mask_list:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (target_W, target_H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            sample_mask_list_resized.append(mask_resized)
        sample_mask_list = np.stack(sample_mask_list_resized, 0)

        # Adjust K matrix for half resolution if provided
        if K is not None:
            if isinstance(K, np.ndarray):
                K = K.copy()
                if K.ndim == 2:  # Single K matrix
                    K[0, 0] /= 2  # fx
                    K[1, 1] /= 2  # fy
                    K[0, 2] /= 2  # cx
                    K[1, 2] /= 2  # cy
                elif K.ndim == 3:  # Per-frame K matrices
                    for t in range(len(K)):
                        K[t, 0, 0] /= 2
                        K[t, 1, 1] /= 2
                        K[t, 0, 2] /= 2
                        K[t, 1, 2] /= 2

        logging.info(
            f"Processing at half resolution: {target_H}x{target_W} (original: {orig_H}x{orig_W})"
        )
    else:
        logging.info(f"Processing at original resolution: {orig_H}x{orig_W}")
        target_H, target_W = orig_H, orig_W

    full_video_pt, full_dep_pt = make_spatracker_input(img_list, dep_list)
    _, T, _, H, W = full_video_pt.shape
    assert sample_mask_list.shape == (T, H, W), f"{sample_mask_list.shape} != {T, H, W}"
    depth_mask = full_dep_pt.squeeze(1) > 1e-6
    logging.info(f"T=[{T}], video shape: {full_video_pt.shape}")

    start_t = time.time()
    # viz the fg mask
    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * full_video_pt[
        0
    ].cpu().permute(0, 2, 3, 1)
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_depth_boundary_mask.mp4"),
        (depth_mask.detach().cpu().float().numpy() * 255).astype(np.uint8),
    )

    video_pt = full_video_pt.clone()
    dep_pt = full_dep_pt.clone()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    tracks, visibility, pred2d = [], [], []
    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))

    for round in range(num_slice):
        logging.info(f"Round {round + 1}/{num_slice} ...")
        masks = sample_mask_list * depth_mask.to(sample_mask_list)
        queries = get_uniform_random_queries(
            video_pt, int(chunk_size * (1.0 - support_ratio)), mask_list=masks
        )
        queries_uniform = get_uniform_random_queries(
            video_pt,
            int(chunk_size * support_ratio),
            mask_list=depth_mask.to(sample_mask_list),
        )
        queries = torch.cat([queries, queries_uniform], 1)

        # Visualize queries at processing resolution
        viz_list = viz_queries(queries.squeeze(0), H, W, T)
        imageio.mimsave(
            osp.join(viz_dir, f"{save_name}_r={round}_quries.mp4"), viz_list
        )

        torch.cuda.empty_cache()
        (
            c2w_traj,
            intrs,
            point_map,
            conf_depth,
            track3d_pred,
            track2d_pred,
            vis_pred,
            conf_pred,
            video,
        ) = model.forward(
            video_pt[0].cpu().numpy(),
            queries=queries[0].cpu().numpy(),
            full_point=False,
            query_no_BA=True,
            stage=1,
        )

        # If we processed at half resolution, scale tracks back to original resolution
        if use_half_res:
            # Scale 2D tracks from processing resolution to original resolution
            track2d_pred[..., 0] *= orig_W / W  # Scale x coordinates
            track2d_pred[..., 1] *= orig_H / H  # Scale y coordinates

            # Note: 3D tracks (track3d_pred) remain in 3D space, no scaling needed
            # Only 2D projections need to be scaled back

        tracks.append(track3d_pred)
        visibility.append(vis_pred.squeeze(-1))
        pred2d.append(track2d_pred)

    tracks = torch.cat(tracks, 1)
    visibility = torch.cat(visibility, 1)
    visibility = visibility > 0.5
    pred2d = torch.cat(pred2d, 1)

    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t) / 60.0:.3f}min")

    # For visualization at original resolution, we need to reconstruct original video
    if use_half_res:
        # Reload original resolution images for visualization
        img_dir = osp.join(working_dir, "images")
        img_fns = sorted(
            [
                it
                for it in os.listdir(img_dir)
                if it.endswith(".png") or it.endswith(".jpg")
            ]
        )
        orig_img_list = [
            imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns
        ]
        orig_img_list = np.asarray(orig_img_list)
        video_pt_orig, _ = make_spatracker_input(
            orig_img_list, dep_list
        )  # dep_list not used for viz

        # Scale tracks back to original resolution for visualization
        viz_tracks = pred2d[None, :, :max_viz_cnt, :2].clone()
        viz_visibility = visibility[None, :, :max_viz_cnt].clone()

        # Visualize at original resolution
        vis.visualize(
            video=video_pt_orig,
            tracks=viz_tracks,
            visibility=viz_visibility,
            filename=f"{save_name}_spatracker2_tap",
        )
    else:
        # Efficient viz at processing resolution
        viz_choice = np.random.choice(
            pred2d.shape[1], min(pred2d.shape[1], max_viz_cnt)
        )
        vis.visualize(
            video=video_pt,
            tracks=pred2d[None, :, viz_choice, :2],
            visibility=visibility[None, :, viz_choice],
            filename=f"{save_name}_spatracker2_tap",
        )

    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_spatracker2_tap.npz"),
        tracks=pred2d.cpu().numpy(),  # Already scaled to original resolution
        visibility=visibility.cpu().numpy(),
        K=K,  # Adjusted K if half resolution was used
        processing_resolution=f"{H}x{W}" if use_half_res else "original",
        original_resolution=f"{orig_H}x{orig_W}",
    )
    return


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser("MoSca-V2 Reconstruction")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--no_viz", action="store_true", help="no viz")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)
    model = get_spatracker2("cuda")
    save_name = "uniform_dep=depth_anything"
    img_dir = osp.join(args.ws, "images")
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    img_list = np.asarray(img_list)
    name = "depth_anything_depth"
    dep_dir = osp.join(args.ws, name)
    if osp.isdir(dep_dir):
        dep_fns = sorted([osp.join(dep_dir, fn) for fn in os.listdir(dep_dir)])
        dep_list = np.stack([np.load(fn)["dep"] for fn in dep_fns], 0)
    else:
        # try the npz
        depth_fn = dep_dir + ".npz"
        assert osp.exists(depth_fn), f"Depth not found in {dep_dir} or {depth_fn}"
        dep_list = np.load(depth_fn)["dep"]
    uniform_sample_list = np.ones_like(dep_list) > 0
    spatracker2_process_folder(
        model=model,
        working_dir=args.ws,
        sample_mask_list=uniform_sample_list,
        save_name=save_name,
        total_n_pts=cfg.n_track_uniform,
        dep_list=dep_list,
        K=None,
        img_list=img_list,
    )
