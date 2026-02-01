from pathlib import Path
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState,RenderTabState
import os.path as osp

from viser import Icon

from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import render
from vis.playback_panel import add_gui_playback_group
from vis.render_panel import populate_render_tab
from vis.utils import draw_tracks_2d_th, get_server
from vis.viewer import DynamicViewer


class Renderer:
    def __init__(
        self,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int,
    ):
        
        self.s_model = s_model
        self.d_model = d_model
        self.device = device

        self.work_dir = work_dir
    
        
        server = get_server(port=port)
        self.viewer = nerfview.Viewer(server=server, render_fn=self.render_fn, mode='rendering')
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.d_model.T,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.viewer.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.viewer.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        # self._render_track_checkbox = server.gui.add_checkbox("Render tracks", False)
        # self._render_track_checkbox.on_update(self.viewer.rerender)

        # tabs = server.gui.add_tab_group()
        # with tabs.add_tab("Render", Icon.CAMERA):
        #     self.render_tab_state = populate_render_tab(
        #         server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
        #     )
        

        # self.tracks_3d = self.model.compute_poses_fg(
        #     #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
        #     torch.arange(self.num_frames, device=self.device),
        #     inds=torch.arange(10, device=self.device),
        # )[0]

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(path, f"photometric_s_model_native_add3.pth")
        ),
        device=device,
        )
        s_model.eval()
        d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(path, f"photometric_d_model_native_add3.pth")
        ),
        device=device,
        )
        d_model.eval()
        print(f"num of nodes: {d_model.M}")
        renderer = Renderer(s_model,d_model, device, *args, **kwargs)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        

        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        if self.viewer is None:
            return np.full((H, W, 3), 255, dtype=np.uint8)
        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self._playback_guis[0].value)
            if not self._canonical_checkbox.value
            else None
        )
        

        gs5 = []
        gs5.append(self.s_model())
        gs5.append(self.d_model(t))

        # * identyfy the visible GS
        render_dict = render(
            gs5,
            H,
            W,
            K=K,
            T_cw=w2c
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # mu_cat = torch.cat([it[0] for it in gs5], 0)
        # dep = visible_render_dict["dep"].detach()[0]
        # mask = visible_render_dict["alpha"].detach()[0] > 0.5
        # back_pts = cams.backproject(cams.homo()[mask], dep[mask])
        # back_pts_world = cams.trans_pts_to_world(working_t, back_pts)
        # dist_sq, nearest_id, _ = knn_points(back_pts_world[None], mu_cat[None], K=K)
        # dist_sq = dist_sq[0, :].reshape(-1)
        # nearest_id = nearest_id[0, :].reshape(-1)
        # valid_nn_mask = dist_sq < (d_model.scf.spatial_unit * 3.0) ** 2
        # nearest_id = nearest_id[valid_nn_mask]
        # visibl_emask = torch.zeros_like(mu_cat[:, 0]).bool()
        # if len(nearest_id) > 0:
        #     visible_mask[nearest_id] = True
        # visible_mask = visible_render_dict["visibility_filter"]

        # gs5_cat = []
        # for i in range(5):
        #     gs5_cat.append(torch.cat([it[i] for it in gs5], 0))
        # new_opa = gs5_cat[-2]
        # new_opa[~visible_mask] = new_opa[~visible_mask] * invisble_opa_factor
        # gs5_cat[-2] = new_opa

        # # convert to gray scale
        # gray_sph = RGB2SH(
        #     torch.mean(
        #         SH2RGB(gs5_cat[-1][~visible_mask, :3]), dim=1, keepdim=True
        #     ).expand(-1, 3)
        # )
        # gray_sph[:, 0] = (
        #     gray_sph[:, 0] * (1.0 - inivisble_red_ratio) + inivisble_red_ratio
        # )
        # gray_sph[:, 1:] = gray_sph[:, 1:] * (1.0 - inivisble_red_ratio) + 0.0
        # pad_sph_dim = s_model()[-1].shape[1]
        # if pad_sph_dim > gray_sph.shape[1]:
        #     gray_sph = F.pad(gray_sph, (0, pad_sph_dim - gray_sph.shape[1], 0, 0))
        # gs5_cat[-1][~visible_mask] = gray_sph

        # # * draw also the current camera frame in the scene
        # add_mu = cams.trans_pts_to_world(working_t, camera_mu)
        # add_fr = (
        #     torch.eye(3).to(add_mu.device).unsqueeze(0).expand(add_mu.shape[0], -1, -1)
        # )
        # add_s = torch.ones_like(add_mu) * 0.001
        # add_o = torch.ones_like(add_s[:, :1]) * 1.0  # 0.4
        # add_sph = torch.ones_like(add_s) * 0.0
        # add_sph[:, 1] = 1.0
        # if pad_sph_dim > add_sph.shape[1]:
        #     add_sph = F.pad(add_sph, (0, pad_sph_dim - add_sph.shape[1], 0, 0))

        # render_dict = render(
        #     [
        #         gs5_cat,
        #         [
        #             add_mu.to(device),
        #             add_fr.to(device),
        #             add_s.to(device),
        #             add_o.to(device),
        #             add_sph.to(device),
        #         ],
        #     ],
        #     H,
        #     W,
        #     K=cams.K(H, W),
        #     T_cw=pose_list[cam_tid],
        # )
        # rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        # img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        

        # if not self.viewer._render_track_checkbox.value:
        #     img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        # else:
        #     assert t is not None
        #     tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
        #     tracks_2d = torch.einsum(
        #         "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
        #     )
        #     tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
        #     img = draw_tracks_2d_th(img, tracks_2d)
        return img