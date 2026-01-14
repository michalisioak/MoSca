import cv2
import matplotlib
import torch
import os, sys
import os.path as osp
import imageio
import numpy as np
from cotracker_visualizer import Visualizer
from tapnet.utils.transforms import convert_grid_coordinates
import torch.nn.functional as F

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

sys.path.append(osp.dirname(osp.abspath(__file__)))
import logging, time
from tracking_utils import get_uniform_random_queries, convert_img_list_to_tapnet_input

tap_size = 256 * 2

@torch.no_grad()
def get_tapnext_model(device,ckpt_path=osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "../../weights/tapnet/bootstapnext_ckpt.npz")
    )):
    model = TAPNext(image_size=(tap_size, tap_size))
    model = restore_model_from_jax_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def tapnext_process_folder(
    working_dir,
    img_list:list,
    sample_mask_list,
    model: TAPNext,
    total_n_pts,
    chunk_size=1000,  # designed for 16GB GPU
    save_name="",
    max_viz_cnt=512,    
    device=torch.device("cuda:0"),
):
    viz_dir = osp.join(working_dir, "tapnext_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,
        tracks_leave_trace=20,
    )

    # * prepare video,
    tap_size = 256
    video_pt, ori_H, ori_W, ori_video_pt = convert_img_list_to_tapnet_input(
        img_list, tap_size, tap_size
    )
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}testing_scaling.mp4"),
        video_pt.cpu().numpy().astype(np.uint8),
    )
    video_pt = video_pt.to(device)
    logging.info(
        f"Loaded video frame: {video_pt.shape} and original size: {ori_H}x{ori_W}"
    )
    # frame size is T,256,256,3 uint8

    # * prepare and resize the dense mask
    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * ori_video_pt.cpu()
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )

    # ! legacy old version may need the sub-sample time, here set the dummy one
    T = len(video_pt)
    logging.info(f"Start tracking...")

    # ! resize the mask to tap_size
    sample_mask_list = (
        F.interpolate(
            sample_mask_list.float()[:, None],
            size=(tap_size, tap_size),
            mode="nearest",
        ).squeeze(1)
        > 0
    )  # T,256,256

    queries = get_uniform_random_queries(
        video_pt[None], total_n_pts, mask_list=sample_mask_list
    ).squeeze(
        0
    )  # N,3 [t,Wind, Hind]
    queries = queries[:, [0, 2, 1]]  # N,3 [t,Hind, Wind]
    queries = queries.to(torch.int32)
    queries[:, 0] = torch.clamp(queries[:, 0], 0, T - 1)
    queries[:, 1] = torch.clamp(queries[:, 1], 0, tap_size - 1)
    queries[:, 2] = torch.clamp(queries[:, 2], 0, tap_size - 1)
    queries = queries.to(device)


    # query: N,3, in the last dim is [t, [0-H=255], [0-W=255]] in int32
    # TODO: chunk wise save memory
    cur = 0
    tracks, visibility = [], []
    while cur < total_n_pts:
        logging.info(f"Processing {cur}-{cur+chunk_size}/{total_n_pts}")
        cur_queries = queries[cur : min(cur + chunk_size, len(queries))]
        _tracks, _visibility = inference(video_pt, cur_queries, model)
        tracks.append(_tracks)
        visibility.append(_visibility)
        cur = cur + chunk_size
    tracks = torch.cat(tracks, dim=0)
    visibility = torch.cat(visibility, dim=0)
    # tracks, visibility = inference(video_pt, queries, model)  # N,T,2; N,T

    # tracks = torch.from_numpy(convert_grid_coordinates(
    #     tracks.cpu().numpy(), (tap_size, tap_size), (ori_W, ori_H)
    # ))
    from tapnet_pt import transforms
    tracks = transforms.convert_grid_coordinates(
        tracks.cpu(), (tap_size, tap_size), (ori_W, ori_H)
    )
    visibility = visibility.cpu()

    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_result.mp4"),
        plot_2d_tracks(ori_video_pt.numpy(), tracks.permute(1,0,2).numpy(), visibility.permute(1,0).numpy()),
    )

    

    tracks = tracks.permute(1, 0, 2).cpu()
    visibility = visibility.permute(1, 0).cpu()

    
    

    viz_choice = np.random.choice(tracks.shape[1], min(tracks.shape[1], max_viz_cnt))
    vis.visualize(
        video=ori_video_pt.permute(0, 3, 1, 2)[None],  # 1,T,3,H,W
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_tapnext_global",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")
    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_tapnext_tap.npz"),
        tracks=tracks.cpu().numpy(),
        visibility=visibility.cpu().numpy(),
    )

def plot_2d_tracks(
    video,
    points,
    visibles,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False,
):
  """Visualize 2D point trajectories."""
  num_frames,num_points  = points.shape[:2]
  print("Plotting 2D tracks:", points.shape, visibles.shape)
  

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1) # type: ignore
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = (np.array(color_map(cmap_norm(i)))[:3] * 255).astype(
        np.uint8
    )

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[
        max(0, t - tracks_leave_trace) : t + 1
    ]
    for s in range(line_tracks.shape[0] - 1):
      img = frame.copy()

      for i in range(num_points):
        if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
        elif (
            show_occ
            and line_infront_cameras[s, i]
            and line_infront_cameras[s + 1, i]
        ):  # occluded
          x1, y1 = int(round(line_tracks[s, i, 0])), int(
              round(line_tracks[s, i, 1])
          )
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(
              round(line_tracks[s + 1, i, 1])
          )
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], -1, cv2.LINE_AA)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 3, point_colors[i], 1, cv2.LINE_AA)

    frames.append(frame)
  
  return frames

@torch.no_grad()
def inference(frames:torch.Tensor, query_points, model:TAPNext,radius=8,threshold=0.5):
    assert frames.dim() == 4, frames.shape
    assert frames.shape[-1] == 3, frames.shape
    # # Normalize frames
    frames = frames.float()
    frames = frames / 255 * 2 - 1

    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    tracks: torch.Tensor
    visibles: torch.Tensor
    tracks, track_logits, visible_logits, tracking_state = model.forward(frames, query_points)

    pred_certainty = tracker_certainty(tracks, track_logits, radius)
    pred_visible_and_certain = (
        F.sigmoid(visible_logits) * pred_certainty
    ) > threshold
    visibles = pred_visible_and_certain.squeeze(-1)
    
    return tracks[0].permute(1, 0, 2), visibles[0].permute(1, 0)  # N,T,2; N,T

if __name__ == "__main__":
    src = "./demo/train"
    img_dir = osp.join(src, "images")
    img_fns = sorted([it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")])
    img_list = [imageio.v2.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    uniform_sample_list = np.ones([len(img_list),img_list[0].shape[0],img_list[0].shape[1]]) > 0
    model = get_tapnext_model("cuda")
    tapnext_process_folder(
        working_dir=src,
        img_list=img_list,
        model=model,
        total_n_pts=2000,
        sample_mask_list=uniform_sample_list,
    )