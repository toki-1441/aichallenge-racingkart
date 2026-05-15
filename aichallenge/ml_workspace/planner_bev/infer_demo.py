#!/usr/bin/env python3
"""Load checkpoint and run forward on one .npz (sanity check)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lib.model import BevTrajectoryNet
from lib.rule_score import select_best_trajectory


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument(
        "--selection",
        type=str,
        default="rule",
        choices=("rule", "teacher"),
        help="teacher uses mode_id in .npz as selected head index",
    )
    args = p.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    bev = torch.from_numpy(np.asarray(z["bev"], dtype=np.float32)).unsqueeze(0)
    bev_np = np.asarray(z["bev"], dtype=np.float32)

    model = BevTrajectoryNet(num_heads=args.num_heads, horizon=args.horizon)
    try:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        pred = model(bev, None).numpy()  # (1,K,T,2)
    pred0 = pred[0]
    if args.selection == "teacher":
        mid = int(np.asarray(z["mode_id"]).ravel()[0])
        if not (0 <= mid < pred0.shape[0]):
            raise ValueError(f"mode_id={mid} out of range for K={pred0.shape[0]}")
        best = mid
    else:
        best = select_best_trajectory(bev_np, pred0)
    print(f"pred shape: {pred.shape}  selection={args.selection}  head: {best}")


if __name__ == "__main__":
    main()
