import torch

from cameras import MonocularCameras


def backproject(uv: torch.Tensor, d: torch.Tensor, cams: MonocularCameras):
    assert uv.ndim == 2
    # uv: always be [-1,+1] on the short side
    assert uv.ndim == d.ndim + 1
    assert uv.shape[-1] == 2
    dep = d[..., None]
    rel_f = torch.as_tensor(cams.rel_focal).to(uv)
    # focal = rel_f / 2.0 * min(H, W)
    cxcy = torch.as_tensor(cams.cxcy_ratio).to(uv) * 2.0 - 1.0
    xy = (uv - cxcy[None, :]) * dep / rel_f[None]
    z = dep
    xyz = torch.cat([xy, z], dim=-1)
    return xyz
