import sys
import os, os.path as osp
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


def sample_frames_every_n(img_list, dep_list, n=10):
    """
    Sample every nth frame from the image and depth lists.

    Args:
        img_list: List/array of images [T, H, W, 3]
        dep_list: List/array of depth maps [T, H, W]
        n: Sample every nth frame

    Returns:
        Sampled frames and their indices
    """
    T = len(img_list)
    indices = list(range(0, T, n))

    # Ensure last frame is included if not already
    if indices[-1] != T - 1:
        indices.append(T - 1)
        indices = sorted(list(set(indices)))

    sampled_imgs = img_list[indices]
    sampled_deps = dep_list[indices]

    return sampled_imgs, sampled_deps, indices


def interpolate_tracks(tracks, visibility, original_indices, target_length):
    """
    Linearly interpolate tracks and visibility to original video length.

    Args:
        tracks: Sampled tracks [N_sampled, N_pts, 3]
        visibility: Sampled visibility [N_sampled, N_pts]
        original_indices: Indices of sampled frames in original video
        target_length: Original video length T

    Returns:
        Interpolated tracks [target_length, N_pts, 3]
        Interpolated visibility [target_length, N_pts]
    """
    import torch
    import numpy as np
    from scipy.interpolate import interp1d

    T_target = target_length
    N_pts = tracks.shape[1]

    # Convert to numpy for interpolation
    tracks_np = tracks.cpu().numpy() if torch.is_tensor(tracks) else tracks
    vis_np = visibility.cpu().numpy() if torch.is_tensor(visibility) else visibility

    # Create interpolation function for each point
    interp_tracks = np.zeros((T_target, N_pts, 3))
    interp_vis = np.zeros((T_target, N_pts))

    for pt_idx in range(N_pts):
        # Interpolate tracks (3D coordinates)
        for coord in range(3):
            # Only interpolate where valid
            valid_mask = ~np.isnan(tracks_np[:, pt_idx, coord])
            if np.sum(valid_mask) >= 2:  # Need at least 2 points for interpolation
                valid_indices = np.array(original_indices)[valid_mask]
                valid_values = tracks_np[valid_mask, pt_idx, coord]

                f = interp1d(
                    valid_indices,
                    valid_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interp_tracks[:, pt_idx, coord] = f(np.arange(T_target))
            else:
                # If insufficient data, use nearest neighbor
                nearest_idx = np.argmin(
                    np.abs(
                        np.arange(T_target)[:, None]
                        - np.array(original_indices)[None, :]
                    ),
                    axis=1,
                )
                interp_tracks[:, pt_idx, coord] = tracks_np[nearest_idx, pt_idx, coord]

        # Interpolate visibility (binary values, use nearest for edges)
        f_vis = interp1d(
            original_indices,
            vis_np[:, pt_idx],
            kind="nearest",  # Use nearest for binary data
            bounds_error=False,
            fill_value=(vis_np[0, pt_idx], vis_np[-1, pt_idx]),
        )
        interp_vis[:, pt_idx] = f_vis(np.arange(T_target))

    # Convert back to tensor if needed
    if torch.is_tensor(tracks):
        interp_tracks = torch.from_numpy(interp_tracks).float()
        interp_vis = torch.from_numpy(interp_vis).float()

    return interp_tracks, interp_vis


@torch.no_grad()
def spatracker2_process_folder(
    working_dir,
    img_list,
    dep_list,
    sample_mask_list,
    model: Predictor,
    total_n_pts,
    K,
    chunk_size=10000,
    save_name="",
    max_viz_cnt=512,
    support_ratio=0.2,
    sample_every_n=1,
):
    """
    Process video by sampling every nth frame and interpolating the rest.
    """
    viz_dir = osp.join(working_dir, "spatracker2_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)

    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,
        tracks_leave_trace=4,
    )

    # Sample frames every nth
    sampled_imgs, sampled_deps, sampled_indices = sample_frames_every_n(
        img_list, dep_list, sample_every_n
    )

    # Also sample the mask accordingly
    sampled_mask = sample_mask_list[sampled_indices]

    logging.info(f"Original video length: {len(img_list)}")
    logging.info(f"Sampled video length: {len(sampled_imgs)}")
    logging.info(f"Sampled indices: {sampled_indices}")

    # Make input for sampled frames
    full_video_pt, full_dep_pt = make_spatracker_input(sampled_imgs, sampled_deps)
    _, T_sampled, _, H, W = full_video_pt.shape

    # vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    # vggt4track_model.eval()
    # vggt4track_model = vggt4track_model.to("cuda")

    # with torch.no_grad():
    #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #         # Predict attributes including cameras, depth maps, and point maps.
    #         predictions = vggt4track_model(full_video_pt.cuda() / 255)
    #         extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
    #         depth_map, depth_conf = (
    #             predictions["points_map"][..., 2],
    #             predictions["unc_metric"],
    #         )

    # depth_tensor = depth_map.squeeze().cpu().numpy()
    # extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    # extrs = extrinsic.squeeze().cpu().numpy()
    # intrs = intrinsic.squeeze().cpu().numpy()
    # unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    # Process with original function on sampled frames
    tracks_sampled, visibility_sampled, pred2d_sampled = [], [], []

    depth_mask = full_dep_pt.squeeze(1) > 1e-6
    sample_mask_tensor = torch.as_tensor(sampled_mask).cpu() > 0

    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))

    for round in range(num_slice):
        logging.info(f"Round {round + 1}/{num_slice} ...")
        masks = sample_mask_tensor * depth_mask.to(sample_mask_tensor)

        queries = get_uniform_random_queries(
            full_video_pt, int(chunk_size * (1.0 - support_ratio)), mask_list=masks
        )
        queries_uniform = get_uniform_random_queries(
            full_video_pt,
            int(chunk_size * support_ratio),
            mask_list=depth_mask.to(sample_mask_tensor),
        )
        queries = torch.cat([queries, queries_uniform], 1)

        torch.cuda.empty_cache()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                full_video_pt[0].cpu().numpy(),
                queries=queries[0].cpu().numpy(),
                full_point=False,
                query_no_BA=True,
                # depth=depth_tensor,
                # intrs=intrs,
                # extrs=extrs,
                # unc_metric=unc_metric,
                stage=1,
                support_frame=len(full_video_pt[0]) - 1,
            )

        tracks_sampled.append(track3d_pred)
        visibility_sampled.append(vis_pred.squeeze(-1))
        pred2d_sampled.append(track2d_pred)

    tracks_sampled = torch.cat(tracks_sampled, 1)
    visibility_sampled = torch.cat(visibility_sampled, 1)
    visibility_sampled = visibility_sampled > 0.5
    pred2d_sampled = torch.cat(pred2d_sampled, 1)

    # ============ INTERPOLATE CAMERA POSES ============
    logging.info("Interpolating camera poses to original video length...")
    T_original = len(img_list)

    # Convert extrinsics to a more interpolation-friendly format
    # extrinsics are T_wc (world-to-camera) matrices
    # camera_poses_sampled = extrs  # Shape: [T_sampled, 4, 4]

    # For interpolation, we'll work with rotation matrices and translations separately
    # rotations_sampled = camera_poses_sampled[:, :3, :3]  # [T_sampled, 3, 3]
    # translations_sampled = camera_poses_sampled[:, :3, 3]  # [T_sampled, 3]

    # Convert rotations to quaternions for smoother interpolation
    # from scipy.spatial.transform import Rotation as R

    # quaternions_sampled = []
    # for i in range(len(rotations_sampled)):
    #     r = R.from_matrix(rotations_sampled[i])
    #     quaternions_sampled.append(r.as_quat())  # [x, y, z, w] format
    # quaternions_sampled = np.array(quaternions_sampled)  # [T_sampled, 4]

    # # Interpolate quaternions and translations
    # from scipy.interpolate import interp1d
    # from scipy.spatial.transform import Slerp

    # quaternions_interpolated = np.zeros((T_original, 4))
    # translations_interpolated = np.zeros((T_original, 3))

    # # Interpolate translations (linear interpolation)
    # for coord in range(3):
    #     f_trans = interp1d(
    #         sampled_indices,
    #         translations_sampled[:, coord],
    #         kind="linear",
    #         bounds_error=False,
    #         fill_value="extrapolate",
    #     )
    #     translations_interpolated[:, coord] = f_trans(np.arange(T_original))

    # # Interpolate rotations using SLERP for smooth rotation interpolation
    # # Create keyframes for SLERP
    # key_rots = R.from_matrix(rotations_sampled)
    # key_times = sampled_indices

    # # Only use SLERP if we have at least 2 keyframes
    # if len(key_times) >= 2:
    #     slerp = Slerp(key_times, key_rots)
    #     quaternions_interpolated = slerp(np.arange(T_original)).as_quat()
    # else:
    #     # Fallback to nearest neighbor
    #     for t in range(T_original):
    #         nearest_idx = np.argmin(np.abs(np.array(sampled_indices) - t))
    #         quaternions_interpolated[t] = quaternions_sampled[nearest_idx]

    # # Convert quaternions back to rotation matrices
    # rotations_interpolated = np.zeros((T_original, 3, 3))
    # for i in range(T_original):
    #     r = R.from_quat(quaternions_interpolated[i])
    #     rotations_interpolated[i] = r.as_matrix()

    # # Reconstruct camera poses
    # camera_poses_interpolated = np.zeros((T_original, 4, 4))
    # camera_poses_interpolated[:, :3, :3] = rotations_interpolated
    # camera_poses_interpolated[:, :3, 3] = translations_interpolated
    # camera_poses_interpolated[:, 3, 3] = 1.0

    # logging.info(f"Interpolated camera poses shape: {camera_poses_interpolated.shape}")

    # ============ INTERPOLATE DEPTH MAPS ============
    logging.info("Interpolating depth maps to original video length...")

    # Prepare depth maps for interpolation
    H, W = sampled_deps[0].shape
    depth_interpolated = np.zeros((T_original, H, W))

    from scipy.interpolate import interp1d

    # Process in chunks of pixels to save memory
    total_pixels = H * W
    chunk_size_pixels = 10000  # Adjust based on your memory

    for start_pixel in range(0, total_pixels, chunk_size_pixels):
        end_pixel = min(start_pixel + chunk_size_pixels, total_pixels)

        logging.info(
            f"Interpolating depth for pixels {start_pixel}-{end_pixel - 1}/{total_pixels}"
        )

        # Process each pixel in this chunk
        for px_idx in range(start_pixel, end_pixel):
            h = px_idx // W
            w = px_idx % W

            # Get depth values for this pixel across sampled frames
            depth_values = sampled_deps[:, h, w]

            # Check if pixel has valid depth in at least some frames
            valid_mask = depth_values > 0
            if np.sum(valid_mask) >= 2:  # Need at least 2 valid points
                valid_indices = np.array(sampled_indices)[valid_mask]
                valid_values = depth_values[valid_mask]

                f = interp1d(
                    valid_indices,
                    valid_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                depth_interpolated[:, h, w] = f(np.arange(T_original))
            else:
                # If insufficient data, use nearest neighbor
                nearest_idx = np.argmin(
                    np.abs(
                        np.arange(T_original)[:, None]
                        - np.array(sampled_indices)[None, :]
                    ),
                    axis=1,
                )
                depth_interpolated[:, h, w] = depth_values[nearest_idx]

    # Ensure non-negative depth values
    depth_interpolated = np.maximum(depth_interpolated, 0)

    logging.info(f"Interpolated depth maps shape: {depth_interpolated.shape}")

    # ============ INTERPOLATE TRACKS ============
    logging.info("Interpolating tracks to original video length...")

    tracks_interpolated, visibility_interpolated = interpolate_tracks(
        tracks_sampled, visibility_sampled, sampled_indices, T_original
    )

    # Also interpolate 2D predictions if needed
    pred2d_interpolated, _ = interpolate_tracks(
        pred2d_sampled, visibility_sampled, sampled_indices, T_original
    )

    logging.info(f"Final interpolated tracks shape: {tracks_interpolated.shape}")

    # ============ SAVE INTERPOLATED CAMERA POSES ============
    camera_output_dir = osp.join(working_dir, "spatracker2_cameras")
    os.makedirs(camera_output_dir, exist_ok=True)

    logging.info(f"Saving interpolated camera poses to {camera_output_dir}")

    # Save in a format compatible with MonocularCameras
    # MonocularCameras expects either:
    # 1. For independent mode: T_wc matrices for each frame
    # 2. For delta mode: T_(i)(i+1) matrices (relative poses)

    # Save as T_wc (world-to-camera) matrices
    camera_file = osp.join(camera_output_dir, f"{save_name}_cameras.npz")
    # np.savez_compressed(
    #     camera_file,
    #     poses=camera_poses_interpolated,  # [T, 4, 4] T_wc matrices
    #     intrinsic=intrs[0]
    #     if len(intrs) > 0
    #     else None,  # Use first intrinsic if constant
    #     sampled_indices=np.array(sampled_indices),
    #     original_length=T_original,
    #     H=H,
    #     W=W,
    # )

    # Also save per-frame camera files (similar to depth format)
    # for frame_idx in range(T_original):
    #     filename = f"camera_{frame_idx:05d}.npz"
    #     filepath = osp.join(camera_output_dir, filename)

    #     np.savez_compressed(
    #         filepath,
    #         pose=camera_poses_interpolated[frame_idx],  # 4x4 T_wc matrix
    #         intrinsic=intrs[0] if len(intrs) > 0 else None,
    #     )

    #     if (frame_idx + 1) % 100 == 0:
    #         logging.info(f"Saved {frame_idx + 1}/{T_original} camera frames")

    # logging.info(
    #     f"Successfully saved {T_original} camera frames to {camera_output_dir}"
    # )

    # # ============ OPTIONAL: Visualize camera trajectory ============
    # viz_camera_dir = osp.join(viz_dir, "camera_trajectory")
    # os.makedirs(viz_camera_dir, exist_ok=True)

    # # Simple visualization: plot camera positions in 3D
    # try:
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D

    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection="3d")

    #     # Plot camera centers (positions)
    #     camera_centers = -np.einsum(
    #         "tij,tj->ti",
    #         camera_poses_interpolated[:, :3, :3].transpose(0, 2, 1),
    #         camera_poses_interpolated[:, :3, 3],
    #     )

    #     ax.plot(
    #         camera_centers[:, 0],
    #         camera_centers[:, 1],
    #         camera_centers[:, 2],
    #         "b-",
    #         alpha=0.7,
    #     )
    #     ax.scatter(
    #         camera_centers[0, 0],
    #         camera_centers[0, 1],
    #         camera_centers[0, 2],
    #         c="g",
    #         s=100,
    #         marker="o",
    #         label="Start",
    #     )
    #     ax.scatter(
    #         camera_centers[-1, 0],
    #         camera_centers[-1, 1],
    #         camera_centers[-1, 2],
    #         c="r",
    #         s=100,
    #         marker="o",
    #         label="End",
    #     )

    #     # Mark sampled frames
    #     sampled_centers = camera_centers[sampled_indices]
    #     ax.scatter(
    #         sampled_centers[:, 0],
    #         sampled_centers[:, 1],
    #         sampled_centers[:, 2],
    #         c="orange",
    #         s=50,
    #         marker="^",
    #         label="Sampled frames",
    #     )

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Camera Trajectory (Interpolated)")
    #     ax.legend()

    #     plt.savefig(
    #         osp.join(viz_camera_dir, "camera_trajectory.png"),
    #         dpi=150,
    #         bbox_inches="tight",
    #     )
    #     plt.close()

    #     logging.info(f"Camera trajectory visualization saved to {viz_camera_dir}")
    # except Exception as e:
    #     logging.warning(f"Could not create camera trajectory visualization: {e}")

    # ============ SAVE DEPTH MAPS IN 0_00000.npz FORMAT ============
    # depth_output_dir = osp.join(working_dir, "spatracker2_depth")
    # os.makedirs(depth_output_dir, exist_ok=True)

    # logging.info(f"Saving interpolated depth maps to {depth_output_dir}")

    # for frame_idx in range(T_original):
    #     # Format: 0_00000.npz, 0_00001.npz, etc.
    #     filename = f"0_{frame_idx:05d}.npz"
    #     filepath = osp.join(depth_output_dir, filename)

    #     # Save depth map for this frame
    #     np.savez_compressed(
    #         filepath,
    #         dep=depth_interpolated[frame_idx],
    #     )

    #     if (frame_idx + 1) % 100 == 0:
    #         logging.info(f"Saved {frame_idx + 1}/{T_original} depth frames")

    # logging.info(f"Successfully saved {T_original} depth frames to {depth_output_dir}")

    # # ============ OPTIONAL: Save a few depth visualization examples ============
    # viz_depth_dir = osp.join(viz_dir, "depth_maps_examples")
    # os.makedirs(viz_depth_dir, exist_ok=True)

    # # Save a few example depth maps as visualizations
    # example_indices = [
    #     0,
    #     T_original // 4,
    #     T_original // 2,
    #     3 * T_original // 4,
    #     T_original - 1,
    # ]
    # for idx in example_indices:
    #     if idx < T_original:
    #         # Normalize depth for visualization
    #         depth_viz = depth_interpolated[idx]
    #         if depth_viz.max() > depth_viz.min():
    #             depth_viz = (depth_viz - depth_viz.min()) / (
    #                 depth_viz.max() - depth_viz.min()
    #             )
    #         depth_viz = (depth_viz * 255).astype(np.uint8)

    #         # Convert to color map for better visualization
    #         depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    #         cv2.imwrite(
    #             osp.join(viz_depth_dir, f"depth_frame_{idx:05d}.png"), depth_viz_color
    #         )

    # # Visualization on interpolated tracks
    viz_choice = np.random.choice(
        pred2d_interpolated.shape[1],
        min(pred2d_interpolated.shape[1], max_viz_cnt),
        replace=False,
    )

    # # Create full video tensor for visualization
    full_video_original = torch.from_numpy(img_list).permute(0, 3, 1, 2).float()[None]

    vis.visualize(
        video=full_video_original,
        tracks=pred2d_interpolated[None, :, viz_choice, :2],
        visibility=visibility_interpolated[None, :, viz_choice],
        filename=f"{save_name}_spatracker2_tap_interpolated",
    )

    # Save tracks results
    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_spatracker2_tap.npz"),
        tracks=pred2d_interpolated.cpu().numpy(),
        visibility=visibility_interpolated.cpu().numpy(),
        sampled_indices=np.array(sampled_indices),
        K=K,
    )

    return (
        tracks_interpolated,
        visibility_interpolated,
        # depth_interpolated,
        # camera_poses_interpolated,
    )


# Modified main section
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser("MoSca-V2 Reconstruction")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--no_viz", action="store_true", help="no viz")
    parser.add_argument(
        "--sample_every_n",
        type=int,
        default=10,
        help="Sample every nth frame and interpolate",
    )
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    model = get_spatracker2("cuda")
    save_name = f"uniform_dep=depth_anything_sampled_{args.sample_every_n}"

    img_dir = osp.join(args.ws, "images")
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    img_list = np.asarray(img_list)

    name = f"depth_anything_depth"
    dep_dir = osp.join(args.ws, name)
    if osp.isdir(dep_dir):
        dep_fns = sorted([osp.join(dep_dir, fn) for fn in os.listdir(dep_dir)])
        dep_list = np.stack([np.load(fn)["dep"] for fn in dep_fns], 0)
    else:
        depth_fn = dep_dir + ".npz"
        assert osp.exists(depth_fn), f"Depth not found in {dep_dir} or {depth_fn}"
        dep_list = np.load(depth_fn)["dep"]

    uniform_sample_list = np.ones_like(dep_list) > 0

    # Use the new interpolated version
    spatracker2_process_folder(
        model=model,
        working_dir=args.ws,
        sample_mask_list=uniform_sample_list,
        save_name=save_name,
        total_n_pts=cfg.n_track_uniform,
        dep_list=dep_list,
        K=None,
        img_list=img_list,
        sample_every_n=args.sample_every_n,
    )
