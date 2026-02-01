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

@torch.no_grad()
def infer_spa_tracker(video:torch.Tensor, dep:torch.Tensor, queries:torch.Tensor, model:Predictor, K:torch.Tensor|None, visiblility_th=0.9,device='cuda'):
    start_t = time.time()
    _, T, _, H, W = video.shape

    torch.cuda.empty_cache()

    assert (
        video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3
    ), "video should have size: 1,T,3,H,W"
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert (
        queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3
    ), "queries should have size: 1,N,3"
    if K is not None:
        K = torch.as_tensor(K).to(device).clone()
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert (
                K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3
            ), "K should have size: T,3,3"
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        K = K[None]  # 1,T,3,3

    video = video.to(device)
    dep = dep.to(device)
    queries = queries.to(device)

    (
            c2w_traj, intrs, point_map, conf_depth,
            pred_tracks, track2d_pred, pred_visibility, conf_pred, video
        )  = model.forward(
        video=video[0].cpu().numpy(),
        queries=queries[0].cpu().numpy(),
        depth=dep.cpu().numpy(),
        intrs=K
    )
    pred_tracks=pred_tracks[..., :3]
    print(f"pred_tracks shape: {pred_tracks.shape}")
    print(f"pred_visibility: {pred_visibility.squeeze(-1).shape}")
    # tracks =(torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], pred_tracks[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]) # T,N,3
    # print(f"tracks shape:{tracks.shape}")
    # print(f"pred vis shape: {pred_visibility.squeeze(-1).shape} ")
    # pred_visibility = pred_visibility[0]

    end_t = time.time()
    print(f"SpaT 2 bi-directional time cost: {(end_t - start_t)/60.0:.3f} min")

    return pred_tracks.cpu(), pred_visibility.squeeze(-1).cpu(), video, track2d_pred.cpu(),video.cpu()

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

    full_video_pt, full_dep_pt = make_spatracker_input(img_list, dep_list)
    _, T, _, H, W = full_video_pt.shape
    assert sample_mask_list.shape == (T, H, W), f"{sample_mask_list.shape} != {T,H,W}"
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

    # vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    # vggt4track_model.eval()
    # vggt4track_model = vggt4track_model.to("cuda")

    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    # print(torch.cuda.memory_summary())
    # print(f"K shape: {K.shape}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    # with torch.no_grad():
    #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #         # Predict attributes including cameras, depth maps, and point maps.
    #         predictions = vggt4track_model(video_pt.cuda()/255)
    #         extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
    #         depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    # depth_tensor = depth_map.squeeze().cpu().numpy()
    # extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    # extrs = extrinsic.squeeze().cpu().numpy()
    # intrs = intrinsic.squeeze().cpu().numpy()
    # unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    tracks, visibility, pred2d = [], [], []
    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))
    for round in range(num_slice):
        logging.info(f"Round {round+1}/{num_slice} ...")
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

        viz_list = viz_queries(queries.squeeze(0), H, W, T)
        imageio.mimsave(
            osp.join(viz_dir, f"{save_name}_r={round}_quries.mp4"), viz_list
        )
        torch.cuda.empty_cache()
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_pt[0].cpu().numpy(), 
                        #   depth=full_dep_pt[0].cpu().numpy(),
                            # intrs=intrs, 
                            # intrs=K[None],
                            # extrs=extrs, 
                            queries=queries[0].cpu().numpy(),
                            full_point=False,
                            query_no_BA=True,
                            stage=1,
                            
                            # unc_metric=unc_metric,
                            )
        tracks.append(track3d_pred)
        visibility.append(vis_pred.squeeze(-1))
        pred2d.append(track2d_pred)
    tracks = torch.cat(tracks, 1)
    visibility = torch.cat(visibility, 1)
    visibility = visibility > 0.5
    pred2d = torch.cat(pred2d,1)

    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")

    # efficient viz
    viz_choice = np.random.choice(pred2d.shape[1], min(pred2d.shape[1], max_viz_cnt))
    # print(f"tracks shape {tracks.shape}")
    # print(f"viz_choice: {viz_choice.shape}, max: {viz_choice.max()}, dtype: {viz_choice.dtype}")
    # print(f"tracks2d: {pred2d.shape}")
    # print(f"visibility: {visibility.shape}")
    # print(f"viz_choice has NaN: {np.isnan(viz_choice).any()}")
    # print(f"viz_choice has Inf: {np.isinf(viz_choice).any()}")
    # print(f"Number of unique indices: {len(np.unique(viz_choice))}")
    # # print(f"viz_choice device: {viz_choice.device}")
    # print(f"track2d_pred device: {pred2d.device}")
    # print(f"visibility device: {visibility.device}")
    assert visibility.shape[1] == pred2d.shape[1] == tracks.shape[1]
    print(f"tracks shape: {tracks.shape}")
    print(f"track value: {tracks[0][0]}")
    # assert False
    vis.visualize(
        video=video_pt,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None,:, viz_choice],
        filename=f"{save_name}_spatracker2_tap",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_spatracker2_tap.npz"),
        tracks=pred2d.cpu().numpy(),
        visibility=visibility.cpu().numpy(),
        K=K,  # also save intrinsic for later use if necessary, but seems because the depth is aligned to input depth, so it is not necessary
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
    save_name=f"uniform_dep=depth_anything"
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
        # try the npz
        depth_fn = dep_dir + ".npz"
        assert osp.exists(depth_fn), f"Depth not found in {dep_dir} or {depth_fn}"
        dep_list = np.load(depth_fn)["dep"]
    uniform_sample_list = np.ones_like(dep_list) > 0
    spatracker2_process_folder(model=model,working_dir=args.ws,sample_mask_list=uniform_sample_list,save_name=save_name,total_n_pts=cfg.n_track_uniform,dep_list=dep_list,K=None,img_list=img_list)