# import glob
# import numpy as np
# import torch
# import os
# import os.path as osp
# from dataclasses import dataclass
# import logging
# import imageio
# from matplotlib import pyplot as plt
# import cv2
# from matplotlib import cm
# from tqdm import tqdm
# from utils3d import quaternion_to_matrix

# from lib_mosca_old.dynamic_gs import RGB2SH
# from monocular_cameras.cameras import MonocularCameras


# @dataclass
# class BundleAdjastmentVizConfig:
#     verbose_n = 300
#     fig_n = 300
#     denser_range = []  # [[0, 10]],  # [[0, 40], [1000, 1040]],
#     denser_interval = 1
#     video_rgb: bool = True

#     text_color: tuple[int, int, int] = (255, 0, 0)
#     border_color: tuple[int, int, int] = (100, 255, 100)
#     background_color: tuple[int, int, int] = (1, 1, 1)


# @torch.no_grad()
# class BundleAdjastmentVizualizer:
#     @torch.no_grad()
#     def __init__(
#         self,
#         ws: str,
#         total_steps: int,
#         rgb: np.ndarray,
#         static_tracks: np.ndarray,
#         cfg: BundleAdjastmentVizConfig | None,
#     ):
#         self.total_steps = total_steps
#         self.cfg = cfg
#         self.rgb = rgb
#         self.static_tracks = static_tracks
#         self.loss_list, self.std_list, self.fovx_list, self.fovy_list = [], [], [], []
#         self.flow_loss_list, self.dep_loss_list, self.dep_corr_loss_list = [], [], []
#         self.cam_rot_loss_list, self.cam_trans_loss_list = [], []
#         if cfg is None:
#             return None
#         self.ws = ws
#         self.viz_dir = osp.join(ws, "viz", "bundle")
#         os.makedirs(self.viz_dir, exist_ok=True)
#         if cfg.video_rgb is not None:
#             logging.info("Viz BA points on each frame...")
#             viz_frames = self.viz_ba_point()
#             imageio.mimsave(osp.join(ws, "BA_points.mp4"), viz_frames)

#     @torch.no_grad()
#     def step(
#         self,
#         step: int,
#         loss: torch.Tensor,
#         s_track_valid_mask_w: torch.Tensor,
#         point_ref: torch.Tensor,
#         cam_rot_loss: torch.Tensor,
#         dep_corr_loss: torch.Tensor,
#         uv_loss: torch.Tensor,
#         dep_loss: torch.Tensor,
#         cam_trans_loss: torch.Tensor,
#         cams: MonocularCameras,
#         param_scale: torch.Tensor,
#     ):
#         if self.cfg is None:
#             return

#         point_ref_mean = (point_ref * s_track_valid_mask_w[:, :, None]).sum(0)
#         std = (point_ref - point_ref_mean[None]).norm(dim=-1, p=2)
#         metric_std = (std * s_track_valid_mask_w).sum(0).mean()
#         self.loss_list.append(loss.item())
#         self.dep_corr_loss_list.append(dep_corr_loss.item())
#         self.flow_loss_list.append(uv_loss.item())
#         self.dep_loss_list.append(dep_loss.item())
#         self.cam_rot_loss_list.append(cam_rot_loss.item())
#         self.cam_trans_loss_list.append(cam_trans_loss.item())
#         self.std_list.append(metric_std.item())
#         fov = cams.fov
#         self.fovx_list.append(float(fov[0]))
#         self.fovy_list.append(float(fov[1]))
#         if step % self.cfg.verbose_n == 0 or step == self.total_steps - 1:
#             logging.info(f"loss={loss:.6f}, fov={cams.fov}")
#             logging.info(f"scale max={param_scale.max()} min={param_scale.min()}")

#         viz_flag = (
#             np.array([step >= r[0] and step <= r[1] for r in self.cfg.denser_range])
#             .any()
#             .item()
#         )
#         viz_flag = viz_flag and step % self.cfg.denser_interval == 0
#         viz_flag = (
#             viz_flag or step % self.cfg.fig_n == 0 or step == self.total_steps - 1
#         )
#         if viz_flag:
#             # viz the 3D aggregation as well as the pcl in 3D!
#             viz_frame = self.viz_global_ba(
#                 point_world=point_ref,
#                 mask=s_track_valid_mask_w,
#                 cams=cams,
#                 error=std,
#                 text=f"Step={step}",
#             )
#             imageio.imsave(
#                 osp.join(self.viz_dir, f"static_scaffold_init_{step:06d}.jpg"),
#                 (viz_frame * 255).astype(np.uint8),
#             )

#     @torch.no_grad()
#     def save(self):
#         make_video_from_pattern(
#             osp.join(self.viz_dir, "static_scaffold_init_*.jpg"),
#             osp.join(self.ws, "static_scaffold_init.mp4"),
#         )

#         if self.total_steps > 0:
#             plt.figure(figsize=(21, 3))
#             for plt_i, plt_pack in enumerate(
#                 [
#                     ("loss", self.loss_list),
#                     ("loss_flow", self.flow_loss_list),
#                     ("loss_dep", self.dep_loss_list),
#                     ("loss_dep_corr", self.dep_corr_loss_list),
#                     ("cam_rot", self.cam_rot_loss_list),
#                     ("cam_trans", self.cam_trans_loss_list),
#                     ("std", self.std_list),
#                     ("fov-x", self.fovx_list),
#                     ("fov-y", self.fovy_list),
#                 ]
#             ):
#                 plt.subplot(1, 9, plt_i + 1)
#                 _ = (
#                     plt.plot(plt_pack[1]),
#                     plt.title(plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"),
#                 )
#                 if plt_pack[0].startswith("loss"):
#                     plt.yscale("log")
#             plt.tight_layout()
#             plt.savefig(osp.join(self.viz_dir, "static_scaffold_init.jpg"))
#             plt.close()

#     def viz_global_ba(
#         self,
#         point_world: torch.Tensor,
#         mask,
#         cams,
#         pts_size=0.001,
#         res=480,
#         error=None,
#         text="",
#     ):
#         assert self.cfg is not None
#         T, M = point_world.shape[:2]
#         device = point_world.device
#         mu = point_world.clone()[mask]
#         sph = RGB2SH(self.rgb[mask])
#         s = torch.ones_like(mu) * pts_size
#         fr = torch.eye(3, device=device).expand(len(mu), -1, -1)
#         o = torch.ones(len(mu), 1, device=device)
#         viz_cam_R = quaternion_to_matrix(cams.q_wc)
#         viz_cam_t = cams.t_wc
#         viz_cam_R, viz_cam_t = cams.Rt_wc_list()
#         viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
#         frame_dict = viz_scene(
#             res,
#             res,
#             viz_cam_R,
#             viz_cam_t,
#             viz_f=viz_f,
#             gs5_param=(mu, fr, s, o, sph),
#             bg_color=self.cfg.background_color,
#             draw_camera_frames=True,
#         )
#         frame = np.concatenate([v for v in frame_dict.values()], 1)
#         # * also do color and error if provided
#         id_color = cm.get_cmap("hsv")(np.arange(M, dtype=np.float32) / M)[:, :3]
#         id_color = torch.from_numpy(id_color).to(device)
#         id_color = id_color[None].expand(T, -1, -1)
#         sph = RGB2SH(id_color[mask])
#         id_frame_dict = viz_scene(
#             res,
#             res,
#             viz_cam_R,
#             viz_cam_t,
#             viz_f=viz_f,
#             gs5_param=(mu, fr, s, o, sph),
#             bg_color=self.cfg.background_color,
#             draw_camera_frames=True,
#         )
#         if error is None:
#             id_frame = np.concatenate([v for v in id_frame_dict.values()], 1)
#             frame = np.concatenate([frame, id_frame], 0)
#         else:
#             # render error as well
#             error = error[mask]
#             error_th = error.max()
#             error_color = (error / (error_th + 1e-9)).detach().cpu().numpy()
#             text = text + f" ErrorVizTh={error_th:.6f}"
#             error_color = cm.get_cmap("viridis")(error_color)[:, :3]
#             error_color = torch.from_numpy(error_color).to(device)
#             sph = RGB2SH(error_color)
#             error_frame_dict = viz_scene(
#                 res,
#                 res,
#                 viz_cam_R,
#                 viz_cam_t,
#                 viz_f=viz_f,
#                 gs5_param=(mu, fr, s, o, sph),
#                 bg_color=self.cfg.background_color,
#                 draw_camera_frames=True,
#             )
#             add_frame = np.concatenate(
#                 [
#                     id_frame_dict["scene_camera_20deg"],
#                     error_frame_dict["scene_camera_20deg"],
#                 ],
#                 1,
#             )
#             frame = np.concatenate([frame, add_frame], 0)
#         # imageio.imsave("./debug/viz.jpg", frame)
#         frame = frame.copy()
#         if len(text) > 0:
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             bottomLeftCornerOfText = (10, 30)
#             fontScale = 1
#             fontColor = (1.0, 1.0, 1.0)
#             lineType = 2
#             cv2.putText(
#                 frame,
#                 text,
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 lineType,
#             )
#         return frame

#     def viz_ba_point(self):
#         # todo: color the points by importance robust weight
#         viz_frames = []
#         for t in tqdm(range(len(self.rgb))):
#             frame_rgb = self.rgb[t].copy()
#             uv = self.static_tracks[t]
#             _viz_valid_mask = self.static_tracks_valid_mask[t]
#             for i in range(len(uv)):
#                 if _viz_valid_mask[i]:
#                     u, v = int(uv[i, 0]), int(uv[i, 1])
#                     if 0 <= u < frame_rgb.shape[1] and 0 <= v < frame_rgb.shape[0]:
#                         _color = (
#                             np.array(cm.get_cmap("hsv")(float(i) / len(uv)))[:3] * 255
#                         )
#                         _color = (int(_color[0]), int(_color[1]), int(_color[2]))
#                         # put a circel with color
#                         cv2.circle(
#                             img=frame_rgb,
#                             center=(u, v),
#                             radius=3,
#                             color=_color,
#                             thickness=1,
#                         )
#             # put total valid num as text valid_mask.sum()
#             cv2.putText(
#                 frame_rgb,
#                 f"Visible BA points: {_viz_valid_mask.sum()}",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2,
#                 cv2.LINE_AA,
#             )
#             viz_frames.append(frame_rgb)
#         return viz_frames


# def make_video_from_pattern(pattern, dst: str):
#     fns = glob.glob(pattern)
#     fns.sort()
#     frames = []
#     for fn in fns:
#         frames.append(imageio.imread(fn))
#     logging.info(f"Saving video to {dst} ...")
#     imageio.mimsave(dst, frames)
#     logging.info("Saved!")
#     return


# @torch.no_grad()
# def viz_scene(
#     H,
#     W,
#     param_cam_R_wc,
#     param_cam_t_wc,
#     model=None,
#     viz_f=40.0,
#     save_name=None,
#     viz_first_n_cam=-1,
#     gs5_param=None,
#     bg_color=[1.0, 1.0, 1.0],
#     draw_camera_frames=False,
#     return_full=False,
# ):
#     cam_viz_mask = None
#     camera_down20_R_wc, camera_down20_t_wc = None, None
#     # auto select viewpoint
#     # manually add the camera viz to to
#     if model is None:
#         assert gs5_param is not None
#         mu_w, fr_w, s, o, sph = gs5_param
#     else:
#         mu_w, fr_w, s, o, sph = model()
#     # add the cameras to the GS
#     if draw_camera_frames:
#         mu_w, fr_w, s, o, sph, cam_viz_mask = add_camera_frame(
#             (mu_w, fr_w, s, o, sph), param_cam_R_wc, param_cam_t_wc, viz_first_n_cam
#         )

#     # * prepare the viz camera
#     # * (1) global scene viz
#     # viz camera set manually
#     global_R_wc, global_t_wc = get_global_viz_cam_Rt(
#         mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
#     )
#     global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
#         mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20
#     )
#     if draw_camera_frames:
#         camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
#             mu_w,
#             param_cam_R_wc,
#             param_cam_t_wc,
#             viz_f,
#             factor=0.5,
#             auto_zoom_mask=cam_viz_mask,
#         )
#         camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
#             mu_w,
#             param_cam_R_wc,
#             param_cam_t_wc,
#             viz_f,
#             20,
#             factor=0.5,
#             auto_zoom_mask=cam_viz_mask,
#         )

#     ret = {}
#     ret_full = {}
#     todo = {  # "scene_global": (global_R_wc, global_t_wc),
#         "scene_global_20deg": (global_down20_R_wc, global_down20_t_wc)
#     }
#     if draw_camera_frames:
#         # todo["scene_camera"] = (camera_R_wc, camera_t_wc)
#         todo["scene_camera_20deg"] = (camera_down20_R_wc, camera_down20_t_wc)
#     for name, Rt in todo.items():
#         viz_cam_R_wc, viz_cam_t_wc = Rt
#         viz_cam_R_cw = viz_cam_R_wc.transpose(1, 0)
#         viz_cam_t_cw = -viz_cam_R_cw @ viz_cam_t_wc
#         viz_mu = torch.einsum("ij,nj->ni", viz_cam_R_cw, mu_w) + viz_cam_t_cw[None]
#         viz_fr = torch.einsum("ij,njk->nik", viz_cam_R_cw, fr_w)

#         pf = viz_f / 2 * min(H, W)
#         assert len(viz_mu) == len(sph)
#         render_dict = render_cam_pcl(
#             viz_mu, viz_fr, s, o, sph, H=H, W=W, fx=pf, bg_color=bg_color
#         )
#         rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
#         ret[name] = rgb
#         if return_full:
#             ret_full[name] = render_dict
#         if save_name is not None:
#             base_name = osp.basename(save_name)
#             dir_name = osp.dirname(save_name)
#             os.makedirs(dir_name, exist_ok=True)
#             save_img = np.clip(ret[name] * 255, 0, 255).astype(np.uint8)
#             imageio.imwrite(osp.join(dir_name, f"{name}_{base_name}.jpg"), save_img)
#     if return_full:
#         return ret, ret_full
#     return ret
