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
from vis.extras import draw_gs_point_line, map_colors
from vis.playback_panel import add_gui_playback_group
from vis.render_panel import populate_render_tab
from vis.utils import draw_tracks_2d_th, get_server

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class Renderer:
    def __init__(
        self,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian,
        device: torch.device,
        work_dir: str,
        port: int,
    ):
        self.s_model = s_model
        self.d_model = d_model
        self.device = device
        self.work_dir = work_dir
        
        server = get_server(port=port)
        
        self._time_folder = server.gui.add_folder("Time")
        self.static_opacity_slider = server.gui.add_slider("Static Opacity", min=0,
        max=1,
        step=1,
        initial_value=1,
        )
        self.dynamic_opacity_slider = server.gui.add_slider("Dynamic Opacity", min=0,
        max=1,
        step=1,
        initial_value=1,
        )
        self.show_nodes = server.gui.add_checkbox("Show Nodes",False)
        self.node_opacity = server.gui.add_slider("Node Opacity",0,1,0.01,1)
        self.node_radius = server.gui.add_slider("Node Radius",0,1,0.01,0.05)
        self.line_n = server.gui.add_slider("Line N",0,100,1,32)
        self.line_opacity = server.gui.add_slider("Line Opacity",0,1,0.01,1)
        self.line_radius = server.gui.add_slider("Line Radius",0,1,0.01,0.05)
        self.viewer = nerfview.Viewer(server=server, render_fn=self.render_fn, mode='rendering')
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.d_model.T,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.viewer.rerender)
           

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
    def render_fn(self, camera_state: CameraState, img_dim: tuple[int,int]):
        W, H = img_dim
        # if False:
        #     W = render_tab_state.render_width
        #     H = render_tab_state.render_height
        # else:
        #     W = render_tab_state.viewer_width
        #     H = render_tab_state.viewer_height
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
        t = int(self._playback_guis[0].value)
        
        gs5 = []
        s_gs5 = list(self.s_model())
        s_gs5[-2] = self.static_opacity_slider.value * s_gs5[-2]
        gs5.append(s_gs5)
        d_gs5 = list(self.d_model(t))
        d_gs5[-2] = self.dynamic_opacity_slider.value * d_gs5[-2]
        gs5.append(d_gs5)

        if self.show_nodes.value:
            node_first = self.d_model.scf._node_xyz[0]
            nodes = (
                torch.ones_like(self.d_model.scf.node_sigma.expand(-1, 3)) * self.node_radius.value
            )
            node_colors = map_colors(node_first.detach().cpu().numpy())
            node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())  
            node_s = torch.clamp(nodes, 1e-8, self.d_model.scf.spatial_unit * 3)
            node_o = torch.ones_like(nodes[:, :1]) * self.node_opacity.value
            node_mu = self.d_model.scf._node_xyz[t]
            node_fr = (
                torch.eye(3)
                .to(node_mu.device)
                .unsqueeze(0)
                .expand(node_mu.shape[0], -1, -1)
            )
            gs5.append([node_mu, node_fr, node_s, node_o, node_sph * 0.3])
            if self.line_n.value > 0:
                scf = self.d_model.scf
                dst_xyz = node_mu[scf.topo_knn_ind]
                src_xyz = node_mu[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
                line_xyz = draw_gs_point_line(
                    src_xyz[scf.topo_knn_mask], dst_xyz[scf.topo_knn_mask], n=self.line_n.value
                ).reshape(-1, 3)
                line_fr = (
                    torch.eye(3)
                    .to(node_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = torch.ones_like(line_xyz) * scf.spatial_unit * self.line_radius.value
                line_o = torch.ones_like(line_s[:, :1]) * self.line_opacity.value
                
                src_sph = node_sph[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
                dst_sph = node_sph[scf.topo_knn_ind]
                l_sph = draw_gs_point_line(
                    src_sph[scf.topo_knn_mask], dst_sph[scf.topo_knn_mask], n=self.line_n.value
                ).reshape(-1, node_sph.shape[-1])
               
                gs5.append([line_xyz, line_fr, line_s, line_o, l_sph])


        render_dict = render(
            gs5,
            H,
            W,
            K=K,
            T_cw=w2c
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        return img