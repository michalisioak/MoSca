import os.path as osp

import imageio
import numpy as np

def load_video(ws:str)->np.ndarray:
    video_path = osp.join(ws, "input.mp4")
    video = imageio.v3.imread(video_path)
    return video