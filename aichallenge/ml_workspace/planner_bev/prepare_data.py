#!/usr/bin/env python3
"""Prepare planner_bev training samples as .npz (synthetic or future rosbag pipeline)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from lib.schema import C_BEV, H_BEV, W_BEV


def _write_sample(
    path: Path,
    horizon: int,
    num_heads: int,
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> None:
    """Random smooth forward trajectory + sparse BEV-like raster."""
    bev = np.zeros((C_BEV, H_BEV, W_BEV), dtype=np.float32)
    # lane-ish band along center
    ic, jc = H_BEV // 2, W_BEV // 2
    bev[0, ic - 3 : ic + 3, :] = 0.8
    bev[0, :, max(0, jc - 4) : jc + 4] = 0.3
    bev[2] = np_rng.random((H_BEV, W_BEV), dtype=np.float32) * 0.05

    t = np.linspace(0.0, 15.0, horizon, dtype=np.float32)
    phase = py_rng.random() * 6.28
    amp = py_rng.random() * 0.4 + 0.05
    xs = t
    ys = (amp * np.sin(0.15 * t + phase)).astype(np.float32)
    traj = np.stack([xs, ys], axis=-1)

    mode_id = np.int64(py_rng.randint(0, num_heads - 1))
    np.savez_compressed(path, bev=bev, traj_gt=traj, mode_id=mode_id)


def cmd_synthetic(args: argparse.Namespace) -> None:
    out_train = Path(args.out_train).expanduser().resolve()
    out_val = Path(args.out_val).expanduser().resolve()
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    for i in range(args.num_train):
        _write_sample(out_train / f"{i:06d}.npz", args.horizon, args.num_heads, rng, np_rng)
    for i in range(args.num_val):
        _write_sample(out_val / f"{i:06d}.npz", args.horizon, args.num_heads, rng, np_rng)
    print(f"[OK] wrote {args.num_train} train + {args.num_val} val npz under {out_train} / {out_val}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic", help="Generate random smooth trajectories for smoke / dev training")
    s.add_argument("--out-train", type=str, required=True)
    s.add_argument("--out-val", type=str, required=True)
    s.add_argument("--num-train", type=int, default=512)
    s.add_argument("--num-val", type=int, default=64)
    s.add_argument("--horizon", type=int, default=40)
    s.add_argument("--num-heads", type=int, default=4, help="K; mode_id uniform in [0, K)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synthetic)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
