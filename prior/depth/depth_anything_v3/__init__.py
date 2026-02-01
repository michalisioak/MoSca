from dataclasses import dataclass
import logging
import cv2
import numpy as np
from prior.depth import DepthModel
from depth_anything_3.api import DepthAnything3


@dataclass
class DepthAnythingV3Config:
    model: str = "depth-anything/DA3METRIC-LARGE"


class DepthAnythingV3(DepthModel):
    def __init__(self, cfg: DepthAnythingV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = DepthAnything3.from_pretrained(cfg.model)
        self.model.eval()

    def forward(self, img_list: np.ndarray):
        "returns depth_list, extrinsics, intrinsics"
        logging.info("Depth-Anything processing...")
        T, H, W, C = img_list.shape
        res = self.model.inference(
            list(img_list),
            ref_view_strategy="middle",
            process_res=max(H, W),
        )
        dep_list_unscaled = res.depth
        dep_list = []
        for dep in dep_list_unscaled:
            dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
            dep_list.append(dep)
        return np.asarray(dep_list), res.extrinsics, res.intrinsics
