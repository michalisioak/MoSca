import torch


def compute_dep_loss(
    target_dep: torch.Tensor,
    pred_dep: torch.Tensor,
    sup_mask: torch.Tensor | None = None,
    st_invariant=True,
):
    # pred_dep = render_dict["dep"][0] / torch.clamp(render_dict["alpha"][0], min=1e-6)
    # ! warning, gof does not need divide alpha
    sup_mask = sup_mask.float() if sup_mask is not None else torch.ones_like(target_dep)
    if st_invariant:
        prior_t = torch.median(target_dep[sup_mask > 0.5])
        pred_t = torch.median(pred_dep[sup_mask > 0.5])
        prior_s = (target_dep[sup_mask > 0.5] - prior_t).abs().mean()
        pred_s = (pred_dep[sup_mask > 0.5] - pred_t).abs().mean()
        prior_dep_norm = (target_dep - prior_t) / prior_s
        pred_dep_norm = (pred_dep - pred_t) / pred_s
    else:
        prior_dep_norm = target_dep
        pred_dep_norm = pred_dep
    loss_dep_i = torch.abs(pred_dep_norm - prior_dep_norm) * sup_mask
    loss_dep = loss_dep_i.sum() / sup_mask.sum()
    return loss_dep
