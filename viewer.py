import threading
import time

import numpy as np
import torch
import tyro
from loguru import logger

from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import render

import viser
import nerfview
import os.path as osp
from lib_render.sh_utils import RGB2SH

from viz_utils import draw_gs_point_line, map_colors
import torch.nn.functional as F


def add_gui_playback_group(
    server: viser.ViserServer,
    num_frames: int,
    min_fps: float = 1.0,
    max_fps: float = 60.0,
    fps_step: float = 0.1,
    initial_fps: float = 10.0,
):
    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=False,
    )
    gui_next_frame = server.gui.add_button("Next Frame")
    gui_prev_frame = server.gui.add_button("Prev Frame")
    gui_playing_pause = server.gui.add_button("Pause")
    gui_playing_pause.visible = False
    gui_playing_resume = server.gui.add_button("Resume")
    gui_framerate = server.gui.add_slider(
        "FPS", min=min_fps, max=max_fps, step=fps_step, initial_value=initial_fps
    )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    def _toggle_gui_playing(_):
        gui_playing_pause.visible = not gui_playing_pause.visible
        gui_playing_resume.visible = not gui_playing_resume.visible
        gui_timestep.disabled = gui_playing_pause.visible
        gui_next_frame.disabled = gui_playing_pause.visible
        gui_prev_frame.disabled = gui_playing_pause.visible

    gui_playing_pause.on_click(_toggle_gui_playing)
    gui_playing_resume.on_click(_toggle_gui_playing)

    # Create a thread to update the timestep indefinitely.
    def _update_timestep():
        while True:
            if gui_playing_pause.visible:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1 / gui_framerate.value)

    threading.Thread(target=_update_timestep, daemon=True).start()

    return (
        gui_timestep,
        gui_next_frame,
        gui_prev_frame,
        gui_playing_pause,
        gui_playing_resume,
        gui_framerate,
    )


def scaffold_viz(
    t: int,
    d_model,
    pad_sph_dim: int,
    node_r1=0.003,
    node_opa1=1.0,
    node_r2=0.020,  # 0.01,
    node_opa2=0.012,
    # line
    line_N=32,
    line_color=[0.7] * 3,
    line_opa=0.05,
    line_r_factor=0.0025,  # 0.08
    line_colorful_flag=True,
):
    gs5 = []
    node_first = d_model.scf._node_xyz[0]
    node_colors = map_colors(node_first.detach().cpu().numpy())
    node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())
    node_sph = RGB2SH(torch.from_numpy(node_colors).to(node_first.device).float())
    if pad_sph_dim > node_sph.shape[1]:
        node_sph = F.pad(node_sph, (0, pad_sph_dim - node_sph.shape[1], 0, 0))

    # node_s1 = d_model.scf.node_sigma.expand(-1, 3) * node_r1_factor  # 0.333  # * 0.05
    node_s1 = (
        torch.ones_like(d_model.scf.node_sigma.expand(-1, 3)) * node_r1
    )  # 0.333  # * 0.05
    node_s1 = torch.clamp(node_s1, 1e-8, d_model.scf.spatial_unit * 3)
    node_o1 = torch.ones_like(node_s1[:, :1]) * node_opa1

    node_s2 = torch.ones_like(node_s1) * node_r2
    # node_o2 = torch.ones_like(node_s2[:, :1]) * 0.003
    node_o2 = torch.ones_like(node_s2[:, :1]) * node_opa2

    line_sph = torch.tensor(line_color).to(node_sph.device).float()[None]
    line_sph = RGB2SH(line_sph)
    if pad_sph_dim > line_sph.shape[1]:
        line_sph = F.pad(line_sph, (0, pad_sph_dim - line_sph.shape[1], 0, 0))
    node_mu = d_model.scf._node_xyz[t]
    node_fr = (
        torch.eye(3).to(node_mu.device).unsqueeze(0).expand(node_mu.shape[0], -1, -1)
    )
    gs5.append([node_mu, node_fr, node_s1, node_o1, node_sph * 0.3])
    gs5.append([node_mu, node_fr, node_s2, node_o2, node_sph])
    ##################################################
    if line_N > 0:
        scf = d_model.scf
        dst_xyz = node_mu[scf.topo_knn_ind]
        src_xyz = node_mu[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
        line_xyz = draw_gs_point_line(
            src_xyz[scf.topo_knn_mask], dst_xyz[scf.topo_knn_mask], n=line_N
        ).reshape(-1, 3)
        line_fr = (
            torch.eye(3)
            .to(node_mu.device)
            .unsqueeze(0)
            .expand(line_xyz.shape[0], -1, -1)
        )
        line_s = torch.ones_like(line_xyz) * scf.spatial_unit * line_r_factor
        line_o = torch.ones_like(line_xyz[:, :1]) * line_opa
        if line_colorful_flag:
            src_sph = node_sph[:, None].expand(-1, scf.topo_knn_ind.shape[1], -1)
            dst_sph = node_sph[scf.topo_knn_ind]
            l_sph = draw_gs_point_line(
                src_sph[scf.topo_knn_mask], dst_sph[scf.topo_knn_mask], n=line_N
            ).reshape(-1, node_sph.shape[-1])
        else:
            l_sph = line_sph.expand(len(line_xyz), -1)

        gs5.append([line_xyz, line_fr, line_s, line_o, l_sph])
    return gs5


def main(ws: str, port=8890, device: str = "cuda"):
    rdevice = torch.device(device)
    server = viser.ViserServer(verbose=False, port=port)

    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(ws, "photometric_s_model_native_add3.pth"), map_location=rdevice
        ),
        device=rdevice,
    ).to(rdevice)
    s_model.eval()
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(ws, "photometric_d_model_native_add3.pth"), map_location=rdevice
        ),
        device=rdevice,
    ).to(rdevice)
    # d_model.set_inference_mode()

    gui_up = server.gui.add_vector3(
        "Up Direction",
        initial_value=(0.0, -1.0, -1.0),
        step=0.01,
    )

    @gui_up.on_update
    def _(_) -> None:
        server.scene.set_up_direction(gui_up.value)

    (
        gui_timestep,
        gui_next_frame,
        gui_prev_frame,
        gui_playing_pause,
        gui_playing_resume,
        gui_framerate,
    ) = add_gui_playback_group(server=server, num_frames=d_model.T)

    scaffold = server.gui.add_checkbox("Show Scaffold", False)

    @torch.inference_mode()
    def render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ):
        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(device)
        )
        t = int(gui_timestep.value)

        gs5 = []
        gs5.append(s_model())
        gs5.append(d_model(t))

        if scaffold.value:
            gs5 += scaffold_viz(t=t, d_model=d_model, pad_sph_dim=gs5[0][-1].shape[1])

        # * identyfy the visible GS
        render_dict = render(gs5, H, W, K=K, T_cw=w2c)
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        return img

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

    gui_timestep.on_update(viewer.rerender)
    scaffold.on_update(viewer.rerender)

    logger.info(f"Starting rendering at http://localhost:{server.get_port()}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
