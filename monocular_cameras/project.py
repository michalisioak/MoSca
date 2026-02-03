import torch

from cameras import MonocularCameras


def project(xyz: torch.Tensor, cams: MonocularCameras, th: float = 1e-5):
    assert xyz.shape[-1] == 3
    xy = xyz[..., :2]
    z = xyz[..., 2:]
    z_close_mask = abs(z) < th
    if z_close_mask.any():
        # logging.warning(
        #     f"Projection may create singularity with a point too close to the camera, detected [{z_close_mask.sum()}] points, clamp it"
        # )
        z_close_mask = z_close_mask.float()
        z = (
            z * (1 - z_close_mask) + (1.0 * th) * z_close_mask
        )  # ! always clamp to positive
        assert not (abs(z) < th).any()
    rel_f = torch.as_tensor(cams.rel_focal).to(xyz)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(xyz) * 2.0 - 1.0
    uv = (xy * rel_f[None] / z) + cxcy[None, :]
    return uv  # [-1,1]
