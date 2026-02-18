from typing import Optional
import numpy as np
import torch
import logging


def init_fc(
    fovxfovycxcy: Optional[tuple[float, float, float, float]],
    K: Optional[torch.Tensor],
    height: float,
    width: float,
):
    if K is None:
        if fovxfovycxcy is None:
            fovxfovycxcy = (53.1, 53.1, 0.5, 0.5)
            logging.warning(
                f"Both fxfycxcy and KHW are None, use default {fovxfovycxcy}"
            )
        rel_focal_x = 1.0 / np.tan(np.deg2rad(fovxfovycxcy[0]) / 2.0)
        rel_focal_y = 1.0 / np.tan(np.deg2rad(fovxfovycxcy[1]) / 2.0)
        rel_focal = torch.Tensor([rel_focal_x, rel_focal_y]).squeeze()
        cxcy_ratio = torch.Tensor([fovxfovycxcy[2], fovxfovycxcy[3]])
    else:
        assert fovxfovycxcy is None
        L = min(height, width)
        rel_focal_x = K[0, 0] / L * 2.0
        rel_focal_y = K[1, 1] / L * 2.0
        cx_ratio = K[0, 2] / width
        cy_ratio = K[1, 2] / height
        rel_focal = torch.Tensor([rel_focal_x, rel_focal_y]).squeeze()
        cxcy_ratio = torch.Tensor([cx_ratio, cy_ratio])
    return rel_focal.float(), cxcy_ratio.float()
