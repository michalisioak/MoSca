import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

import os, os.path as osp
import imageio

tap_size = 256

def plot_2d_tracks(
    video,
    points,
    visibles,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False,
):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

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

# Load model
model = TAPNext(image_size=(tap_size, tap_size))
model = restore_model_from_jax_checkpoint(model, "/MoSca/weights/tapnet/bootstapnext_ckpt.npz")
model.to("cuda")
model.eval()

# Load and preprocess frames
src = "./demo/hand"
img_dir = osp.join(src, "images")
img_fns = sorted([it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")])
frames_np = np.array([cv2.resize(imageio.imread(osp.join(img_dir, it))[..., :3], (tap_size, tap_size)) for it in img_fns])
print("Loaded frames:", frames_np.shape)

# Convert to torch tensor and normalize
# Note: TAPNext typically expects input in range [0, 1] or [-1, 1]
# Check the model's expected normalization
frames_tensor = torch.from_numpy(frames_np).float().to("cuda")
frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]
# Or if the model expects [-1, 1]:
# frames_tensor = frames_tensor / 127.5 - 1.0
print("Preprocessed frames tensor:", frames_tensor.shape)
# Add batch dimension and reorder to (B, T, C, H, W)
frames_tensor = frames_tensor.permute(0, 2, 1,3).unsqueeze(0)  # (1, T, C, H, W)
print("Final frames tensor shape:", frames_tensor.shape)

# Create query points
ys, xs = np.meshgrid(np.linspace(8, 256, 8), np.linspace(8, 256, 8))
query_points = np.stack(
    [np.zeros(len(xs.flatten())), xs.flatten(), ys.flatten()], axis=1
)[None]

# Convert query points to torch tensor
query_points_tensor = torch.from_numpy(query_points).float().to("cuda")

# Run model inference
with torch.no_grad():
    pred_tracks, track_logits, pred_visible, tracking_state = model.forward(
        frames_tensor, query_points_tensor
    )

# Convert outputs back to numpy for visualization
pred_tracks_np = pred_tracks.cpu().numpy()
pred_visible_np = pred_visible.cpu().numpy()
print("Predicted tracks shape:", pred_tracks_np[0].shape)
print("Predicted visibility shape:", pred_visible_np[0][...,0].shape)
# # Prepare for visualization
# tracks = [pred_tracks_np]
# visibles = [pred_visible_np]
# tracks = np.stack(tracks, axis=2)[..., ::-1]
# print("Tracks shape for visualization:", tracks.shape)
# visibles = np.stack(visibles, axis=2).squeeze(-1)

# Prepare frames for visualization (denormalize if needed)
print("frames_np dtype before denorm:", frames_np.dtype,frames_np.shape)
# frames_viz = frames_np.astype(np.uint8)
video_viz = plot_2d_tracks(
    frames_np, pred_tracks_np[0], pred_visible_np[0][...,0]
)

# video_viz_uint8 = [frame.astype(np.uint8) for frame in video_viz]

# Save result
imageio.mimsave("./result.mp4", video_viz, fps=30)
print("Video saved to ./result.mp4")