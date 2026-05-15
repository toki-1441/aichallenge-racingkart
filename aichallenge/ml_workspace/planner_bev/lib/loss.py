"""P1 losses: hard-assigned Huber pose, optional diversity & curvature."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pose_loss_hard_assignment(
    pred: torch.Tensor,
    traj_gt: torch.Tensor,
    mode_id: torch.Tensor,
    huber_beta: float = 0.1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        pred: (B, K, T, 2)
        traj_gt: (B, T, 2)
        mode_id: (B,) long in [0, K)
        mask: optional (B,) float/bool — weight per batch row
    """
    b, k, t, _ = pred.shape
    if traj_gt.shape != (b, t, 2):
        raise ValueError(f"traj_gt shape mismatch: {traj_gt.shape} vs ({b},{t},2)")
    if mode_id.shape != (b,):
        raise ValueError(f"mode_id shape mismatch: {mode_id.shape}")
    idx = torch.arange(b, device=pred.device, dtype=torch.long)
    selected = pred[idx, mode_id.long()]  # (B, T, 2)
    per = F.smooth_l1_loss(selected, traj_gt, beta=huber_beta, reduction="none").mean(
        dim=(-1, -2)
    )  # (B,)
    if mask is not None:
        m = mask.float().clamp(0.0, 1.0)
        denom = m.sum().clamp_min(1.0)
        return (per * m).sum() / denom
    return per.mean()


def pose_loss_all_heads_mean(
    pred: torch.Tensor,
    traj_gt: torch.Tensor,
    huber_beta: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over heads of smooth_l1(each head, traj_gt) — weak supervision for non-teacher heads."""
    b, k, t, _ = pred.shape
    if traj_gt.shape != (b, t, 2):
        raise ValueError(f"traj_gt shape mismatch: {traj_gt.shape} vs ({b},{t},2)")
    acc = pred.new_zeros(())
    for ki in range(k):
        sel = pred[:, ki, :, :]
        per = F.smooth_l1_loss(sel, traj_gt, beta=huber_beta, reduction="none").mean(dim=(-1, -2))
        if mask is not None:
            m = mask.float().clamp(0.0, 1.0)
            denom = m.sum().clamp_min(1.0)
            acc = acc + (per * m).sum() / denom
        else:
            acc = acc + per.mean()
    return acc / max(k, 1)


def diversity_loss(pred: torch.Tensor, d_min: float, lam: float) -> torch.Tensor:
    """Encourage pairwise head predictions to differ by at least d_min (meters)."""
    if lam == 0.0:
        return pred.new_zeros(())
    b, k, t, _ = pred.shape
    if k < 2:
        return pred.new_zeros(())
    total = pred.new_zeros(())
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            dist = torch.norm(pred[:, i] - pred[:, j], dim=-1)  # (B, T)
            hinge = torch.clamp(d_min - dist, min=0.0).pow(2).mean()
            total = total + hinge
            count += 1
    return lam * total / float(count)


def curvature_proxy_loss(pred: torch.Tensor, lam: float) -> torch.Tensor:
    """Second finite difference along time (discrete curvature proxy)."""
    if lam == 0.0:
        return pred.new_zeros(())
    if pred.shape[2] < 3:
        return pred.new_zeros(())
    p = pred
    d2 = p[:, :, 2:] + p[:, :, :-2] - 2.0 * p[:, :, 1:-1]
    return lam * d2.pow(2).mean()


def p1_total_loss(
    pred: torch.Tensor,
    traj_gt: torch.Tensor,
    mode_id: torch.Tensor,
    huber_beta: float,
    div_lambda: float,
    div_d_min: float,
    curv_lambda: float,
    mask: torch.Tensor | None = None,
    aux_all_heads_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (total, component dict for logging)."""
    l_pose = pose_loss_hard_assignment(pred, traj_gt, mode_id, huber_beta, mask)
    l_aux = (
        pred.new_zeros(())
        if aux_all_heads_lambda <= 0.0
        else aux_all_heads_lambda * pose_loss_all_heads_mean(pred, traj_gt, huber_beta, mask)
    )
    l_div = diversity_loss(pred, div_d_min, div_lambda)
    l_curv = curvature_proxy_loss(pred, curv_lambda)
    total = l_pose + l_aux + l_div + l_curv
    out = {
        "loss_pose": float(l_pose.detach()),
        "loss_aux_all": float(l_aux.detach()),
        "loss_div": float(l_div.detach()),
        "loss_curv": float(l_curv.detach()),
        "loss_total": float(total.detach()),
    }
    return total, out
