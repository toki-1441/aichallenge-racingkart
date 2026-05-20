#!/usr/bin/env python3
"""Run trained model on a subset of val .npz; overlay trajectories on BEV and save PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from lib.model import BevTrajectoryNet
from lib.rule_score import BEVGridSpec, select_best_trajectory
from lib.schema import H_BEV, W_BEV

_SPEC = BEVGridSpec()


def xy_to_rc(xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,2) ego meters -> integer (row, col) in BEV H×W."""
    r = np.zeros(len(xy_m), dtype=np.int32)
    c = np.zeros(len(xy_m), dtype=np.int32)
    for i in range(len(xy_m)):
        ri, ci = _SPEC.xy_to_rc(float(xy_m[i, 0]), float(xy_m[i, 1]))
        r[i] = np.clip(ri, 0, H_BEV - 1)
        c[i] = np.clip(ci, 0, W_BEV - 1)
    return r, c


def bev_background_rgb(bev: np.ndarray) -> np.ndarray:
    """(4,H,W) in [0,1] -> (H,W,3) uint8: R=ref traj ch, G=obstacles, B=lane."""
    lane = np.clip(bev[0], 0.0, 1.0)
    traj_ch = np.clip(bev[1], 0.0, 1.0)
    obs = np.clip(bev[2], 0.0, 1.0)
    ego = np.clip(bev[3], 0.0, 1.0)
    r = np.clip(traj_ch * 0.9 + ego * 0.35, 0.0, 1.0)
    g = np.clip(obs * 0.95 + ego * 0.2, 0.0, 1.0)
    b = np.clip(lane * 0.9 + ego * 0.25, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _draw_thick_line(
    img: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: tuple[int, int, int], th: int
) -> None:
    h, w = img.shape[:2]
    n = int(max(abs(r1 - r0), abs(c1 - c0), 1) * 2) + 1
    for i in range(n + 1):
        t = i / n
        r = int(round(r0 * (1.0 - t) + r1 * t))
        c = int(round(c0 * (1.0 - t) + c1 * t))
        for dr in range(-th, th + 1):
            for dc in range(-th, th + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    img[rr, cc] = np.array(color, dtype=np.uint8)


def draw_polyline_bev(
    img: np.ndarray, xy_m: np.ndarray, color: tuple[int, int, int], th: int = 2
) -> None:
    if xy_m.shape[0] < 2:
        return
    r, c = xy_to_rc(xy_m.astype(np.float64))
    for t in range(len(r) - 1):
        _draw_thick_line(img, int(r[t]), int(c[t]), int(r[t + 1]), int(c[t + 1]), color, th)


def load_model(cfg_path: Path, ckpt_path: Path, device: torch.device) -> BevTrajectoryNet:
    cfg = OmegaConf.load(cfg_path)
    m = cfg.model
    stem = tuple(int(x) for x in m.stem_channels)
    model = BevTrajectoryNet(
        h_bev=int(m.h_bev),
        w_bev=int(m.w_bev),
        c_in=int(m.c_in),
        aux_dim=int(m.aux_dim),
        num_heads=int(m.num_heads),
        horizon=int(m.horizon),
        stem_channels=stem,
        embed_dim=int(m.embed_dim),
        dropout=float(m.dropout),
    )
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--config", type=str, default="config/train.yaml", help="Hydra YAML for model.*")
    p.add_argument("--val-dir", type=str, default=None, help="Default: data.val_dir from config")
    p.add_argument("--out-dir", type=str, default="viz_val_infer", help="Output directory for PNGs")
    p.add_argument("--num-samples", type=int, default=12, help="Max number of val .npz files to visualize")
    p.add_argument("--seed", type=int, default=0, help="Shuffle val files before taking subset")
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda (aliases: gpu → cuda if available)")
    p.add_argument(
        "--selection",
        type=str,
        default="rule",
        choices=("rule", "teacher"),
        help="rule: lib.rule_score select_best_trajectory; teacher: use mode_id from .npz (debug / upper bound)",
    )
    args = p.parse_args()

    dev = args.device.strip().lower()
    if dev in ("gpu", "cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print(f"[viz_val_infer_bev] WARN: --device {args.device!r} but CUDA unavailable; using cpu")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(cfg_path)
    val_dir = Path(args.val_dir or cfg.data.val_dir).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(val_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {val_dir}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    files = files[: max(1, args.num_samples)]

    model = load_model(cfg_path, ckpt_path, device)
    k_heads = int(cfg.model.num_heads)

    cmap = plt.get_cmap("tab10")
    for path in files:
        z = np.load(path, allow_pickle=False)
        bev = np.asarray(z["bev"], dtype=np.float32)
        traj_gt = np.asarray(z["traj_gt"], dtype=np.float32)
        mode_id = int(np.asarray(z["mode_id"]).ravel()[0])

        bev_t = torch.from_numpy(bev).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(bev_t, None).cpu().numpy()[0]  # (K,T,2)

        if args.selection == "teacher":
            best_head = int(mode_id)
            sel_note = f"teacher pick (mode_id={best_head})"
        else:
            best_head = int(select_best_trajectory(bev, pred))
            sel_note = f"rule pick yellow (head {best_head})"

        img = bev_background_rgb(bev)
        img = img.copy()
        for ki in range(k_heads):
            col = tuple(int(round(c * 255)) for c in cmap(ki % 10)[:3])
            col = tuple(int(c * 0.42) for c in col)
            draw_polyline_bev(img, pred[ki], col, th=1)
        draw_polyline_bev(img, pred[mode_id], (120, 200, 255), th=2)
        draw_polyline_bev(img, traj_gt, (80, 255, 80), th=3)
        draw_polyline_bev(img, pred[best_head], (255, 220, 60), th=3)

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0]})
        ax0, ax1 = axes
        ax0.imshow(img, origin="upper")
        ax0.set_title(
            f"{path.name}\n"
            f"GT green | teacher-head pred cyan (mode_id={mode_id}) | "
            f"all heads faint | {sel_note}",
            fontsize=9,
        )
        ax0.set_xlabel("col (y index)")
        ax0.set_ylabel("row (x index)")

        gt = traj_gt
        ax1.plot(gt[:, 0], gt[:, 1], "g-", linewidth=2.5, label="gt", zorder=10)
        for ki in range(k_heads):
            alpha = 0.95 if ki == best_head else 0.35
            lw = 2.2 if ki == best_head else 1.0
            ax1.plot(
                pred[ki, :, 0],
                pred[ki, :, 1],
                color=cmap(ki % 10),
                alpha=alpha,
                linewidth=lw,
                label=f"head {ki}" + (" (selected)" if ki == best_head else ""),
            )
        ax1.scatter([gt[0, 0]], [gt[0, 1]], c="green", s=70, zorder=11)
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("x forward (m)")
        ax1.set_ylabel("y left (m)")
        ax1.legend(loc="best", fontsize=7)
        ax1.set_title("ego-frame trajectories")

        fig.suptitle(f"checkpoint: {ckpt_path.name}", fontsize=10)
        fig.tight_layout()
        out_path = out_dir / f"{path.stem}_infer_bev.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
