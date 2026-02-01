import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from loguru import logger as guru

from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import render
from vis.renderer import Renderer

import yaml
import viser
import nerfview
import os.path as osp




@dataclass
class RenderConfig:
    work_dir: str
    port: int = 8890


def main(cfg: RenderConfig):
    device = torch.device("cuda")

    renderer = Renderer.init_from_checkpoint(
        cfg.work_dir,
        device,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )

    # guru.info(f"Loading checkpoint from {cfg.work_dir}")
    # s_model = StaticGaussian.load_from_ckpt(
    # torch.load(
    #     osp.join(cfg.work_dir, f"photometric_s_model_native_add3.pth")
    # ),
    # device=device,
    # )
    # s_model.eval()
    # d_model = DynSCFGaussian.load_from_ckpt(
    # torch.load(
    #     osp.join(cfg.work_dir, f"photometric_d_model_native_add3.pth")
    # ),
    # device=device,
    # )
    # d_model.eval()
    # print(f"num of nodes: {d_model.M}")

    # server = viser.ViserServer(verbose=True,port=cfg.port)
    # viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode='rendering')

    guru.info(f"Starting rendering")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))