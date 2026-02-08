import torch


def compute_rgb_loss(
    gt_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    sup_mask: torch.Tensor | None = None,
):
    gt_rgb = gt_rgb.detach()
    sup_mask = sup_mask.float() if sup_mask is not None else torch.ones_like(gt_rgb)
    rgb_loss_i = torch.abs(pred_rgb - gt_rgb.detach()) * sup_mask[..., None]
    rgb_loss = rgb_loss_i.sum() / sup_mask.sum()
    return rgb_loss
