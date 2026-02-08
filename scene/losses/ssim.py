import torch
import kornia


def compute_ssim_loss(
    gt_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    sup_mask: torch.Tensor | None = None,
    window_size: int = 11,
    max_val: float = 1.0,
):
    """
    Compute SSIM loss between predicted and ground truth RGB images with mask support.

    Args:
        gt_rgb: Ground truth RGB tensor of shape (H, W, 3)
        pred_rgb: Predicted RGB tensor of shape (H, W, 3)
        sup_mask: Mask tensor of shape (H, W) with values in [0, 1]
        window_size: Size of the sliding window for SSIM computation
        max_val: Maximum value of the images (typically 1.0 or 255)

    Returns:
        Mean SSIM loss (1 - SSIM) over masked region
    """
    sup_mask = sup_mask.float() if sup_mask is not None else torch.ones_like(gt_rgb)

    # Prepare tensors for kornia (B, C, H, W format)
    pred_batch = pred_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    gt_batch = gt_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Create mask in same format
    mask_batch = sup_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_batch = mask_batch.repeat(1, 3, 1, 1)  # (1, 3, H, W)

    # Apply mask
    pred_masked = pred_batch * mask_batch
    gt_masked = gt_batch * mask_batch

    # Compute SSIM
    ssim_map = kornia.metrics.ssim(
        pred_masked,
        gt_masked,
        window_size=window_size,
        max_val=max_val,
    )  # (1, 3, H, W)

    # Average over channels
    ssim_map = ssim_map.mean(dim=1, keepdim=True)  # (1, 1, H, W)

    # Convert to loss and compute mean only over masked region
    ssim_loss_map = 1.0 - ssim_map
    valid_pixels = (mask_batch[:, :1, ...] > 0).float()

    if valid_pixels.sum() > 0:
        ssim_loss = (ssim_loss_map * valid_pixels).sum() / valid_pixels.sum()
    else:
        ssim_loss = torch.tensor(0.0, device=ssim_loss_map.device)

    return ssim_loss
