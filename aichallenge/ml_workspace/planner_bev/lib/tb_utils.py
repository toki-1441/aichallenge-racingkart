"""TensorBoard helpers for planner_bev training (images, figures, histograms)."""

from __future__ import annotations

import math
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from torch.utils.tensorboard import SummaryWriter

BEV_CHANNEL_NAMES = ("lane", "trajectory", "obstacles", "ego")


def _patch_torch_tensorboard_histogram_numpy() -> None:
    """Torch vendors TB summary code that calls np.greater(..., dtype=...) — invalid in NumPy ≥1.24."""
    import torch.utils.tensorboard.summary as tbs

    if getattr(tbs, "_planner_bev_histogram_patched", False):
        return

    def make_histogram(values, bins, max_bins=None):
        if values.size == 0:
            raise ValueError("The input has no element.")
        values = values.reshape(-1)
        counts, limits = np.histogram(values, bins=bins)
        num_bins = len(counts)
        if max_bins is not None and num_bins > max_bins:
            subsampling = num_bins // max_bins
            subsampling_remainder = num_bins % subsampling
            if subsampling_remainder != 0:
                counts = np.pad(
                    counts,
                    pad_width=[[0, subsampling - subsampling_remainder]],
                    mode="constant",
                    constant_values=0,
                )
            counts = counts.reshape(-1, subsampling).sum(axis=-1)
            new_limits = np.empty((counts.size + 1,), limits.dtype)
            new_limits[:-1] = limits[:-1:subsampling]
            new_limits[-1] = limits[-1]
            limits = new_limits

        cum_counts = np.cumsum((counts > 0).astype(np.int32))
        start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
        start = int(start)
        end = int(end) + 1
        del cum_counts

        counts = counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
        limits = limits[start : end + 1]

        if counts.size == 0 or limits.size == 0:
            raise ValueError("The histogram is empty, please file a bug report.")

        sum_sq = values.dot(values)
        return HistogramProto(
            min=values.min(),
            max=values.max(),
            num=len(values),
            sum=values.sum(),
            sum_squares=sum_sq,
            bucket_limit=limits.tolist(),
            bucket=counts.tolist(),
        )

    tbs.make_histogram = make_histogram  # type: ignore[assignment]
    setattr(tbs, "_planner_bev_histogram_patched", True)


_patch_torch_tensorboard_histogram_numpy()


def _histogram_flat_float64(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Flatten to float64 ndarray for TensorBoard histograms (NumPy 2.x / TB compat)."""
    if isinstance(values, torch.Tensor):
        v = values.detach().float().cpu().numpy().ravel()
    else:
        v = np.asarray(values, dtype=np.float64).ravel()
    return np.asarray(v, dtype=np.float64)


def _downsample_hw(
    x: torch.Tensor, max_side: int
) -> torch.Tensor:
    """x: (C,H,W) float. Bilinear resize if larger than max_side."""
    c, h, w = x.shape
    m = max(h, w)
    if m <= max_side:
        return x
    scale = max_side / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    t = x.unsqueeze(0)  # 1CHW
    y = torch.nn.functional.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    return y.squeeze(0)


def log_bev_channels(
    writer: SummaryWriter,
    bev_chw: torch.Tensor,
    global_step: int,
    tag_prefix: str = "viz/bev",
    max_side: int = 400,
) -> None:
    """Log each BEV channel as a 3-channel grayscale image (CHW for TB)."""
    if bev_chw.dim() != 3:
        return
    x = bev_chw.detach().float().cpu()
    x = torch.clamp(x, 0.0, 1.0)
    x = _downsample_hw(x, max_side)
    c, _, _ = x.shape
    names = list(BEV_CHANNEL_NAMES) + [f"ch{i}" for i in range(len(BEV_CHANNEL_NAMES), c)]
    for i in range(c):
        ch = x[i : i + 1]  # 1,H,W
        rgb = ch.expand(3, -1, -1).clone()
        writer.add_image(f"{tag_prefix}/{names[i]}", rgb, global_step, dataformats="CHW")


def log_trajectory_figure(
    writer: SummaryWriter,
    traj_gt: np.ndarray | torch.Tensor,
    pred_heads: np.ndarray | torch.Tensor,
    mode_id: int,
    global_step: int,
    tag: str = "viz/trajectory_xy",
    title: str = "",
) -> None:
    """
    traj_gt: (T, 2), pred_heads: (K, T, 2), mode_id: assigned teacher head index.
    """
    gt = np.asarray(traj_gt, dtype=np.float64)
    pr = np.asarray(pred_heads, dtype=np.float64)
    k, t, _ = pr.shape
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.5))
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2.5, label="gt", zorder=10)
    cmap = plt.get_cmap("tab10")
    for ki in range(k):
        alpha = 0.95 if ki == int(mode_id) else 0.35
        lw = 2.2 if ki == int(mode_id) else 1.0
        z = 9 if ki == int(mode_id) else 1
        ax.plot(
            pr[ki, :, 0],
            pr[ki, :, 1],
            color=cmap(ki % 10),
            alpha=alpha,
            linewidth=lw,
            label=f"head {ki}" + (" (teacher)" if ki == int(mode_id) else ""),
            zorder=z,
        )
    ax.scatter([gt[0, 0]], [gt[0, 1]], c="green", s=80, marker="o", zorder=11, label="start")
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], c="red", s=80, marker="s", zorder=11, label="end gt")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.legend(loc="best", fontsize=8)
    ttl = title or f"step={global_step}  teacher_head={mode_id}"
    ax.set_title(ttl)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)


def log_model_weights_histograms(
    writer: SummaryWriter,
    model: nn.Module,
    global_step: int,
    name_filter: Optional[List[str]] = None,
) -> None:
    """Log weight histograms for parameters whose name contains any filter substring."""
    n_logged = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.data is None:
            continue
        if name_filter and not any(s in name for s in name_filter):
            continue
        writer.add_histogram(f"weights/{name}", _histogram_flat_float64(p.data), global_step)
        n_logged += 1
    if n_logged == 0:
        for name, p in model.named_parameters():
            if p.requires_grad and p.data is not None:
                writer.add_histogram(f"weights/{name}", _histogram_flat_float64(p.data), global_step)
                break


def log_gradients_histograms(
    writer: SummaryWriter,
    model: nn.Module,
    global_step: int,
    name_filter: Optional[List[str]] = None,
) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name_filter and not any(s in name for s in name_filter):
            continue
        writer.add_histogram(f"grads/{name}", _histogram_flat_float64(p.grad), global_step)


def total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().data.float()
        total += float(g.norm(2).item() ** 2)
    return math.sqrt(total) if total > 0 else 0.0


def log_config_text(writer: SummaryWriter, cfg_yaml: str, step: int = 0) -> None:
    writer.add_text("config/full_yaml", f"```yaml\n{cfg_yaml}\n```", step)


def val_pose_pointwise_l2(
    pred: torch.Tensor, traj_gt: torch.Tensor, mode_id: torch.Tensor
) -> torch.Tensor:
    """Mean L2 over (B,T) for teacher-selected head. pred (B,K,T,2)."""
    b, k, t, _ = pred.shape
    idx = torch.arange(b, device=pred.device, dtype=torch.long)
    sel = pred[idx, mode_id.long()]
    err = torch.norm(sel - traj_gt, dim=-1)  # B,T
    return err.mean(dim=1)  # B
