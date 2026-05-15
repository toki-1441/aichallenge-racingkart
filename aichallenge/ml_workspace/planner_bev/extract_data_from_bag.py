#!/usr/bin/env python3
"""
Extract (bev, traj_gt, mode_id) samples from a rosbag2 directory into .npz files.

Requires: pip install rosbags (see requirements.txt)

Example:
  python3 extract_data_from_bag.py \\
    --bag datasets/rosbag2_planner/planner_20260514_145951 \\
    --out-train datasets/from_bag/train \\
    --out-val datasets/from_bag/val

Then train:
  python3 train.py data.train_dir=datasets/from_bag/train data.val_dir=datasets/from_bag/val

Optional mode_id labeling (see design_docs/planner_mode_id_labeling.md):
  --mode-label kmeans|angular|singleton

Teacher path source (default: MPC prediction markers, high rate):
  --traj-topic /mpc/prediction
  Legacy CSV trajectory (low rate): --traj-topic /planning/scenario_planning/trajectory
  No trajectory topic in bag: --traj-source odom_extrap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lib.extract_from_bag import DEFAULT_TRAJ_TOPIC, extract_samples, write_npz_split


def _clear_npz(d: Path) -> None:
    if not d.exists():
        return
    for p in d.glob("*.npz"):
        p.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", type=str, required=True, help="Path to rosbag2 directory (contains metadata.yaml)")
    p.add_argument("--out-train", type=str, required=True)
    p.add_argument("--out-val", type=str, required=True)
    p.add_argument("--horizon", type=int, default=40, help="Trajectory length T (match model.horizon in train.yaml)")
    p.add_argument("--stride", type=int, default=2, help="Use every N-th BEV message")
    p.add_argument(
        "--sync-slop-ms",
        type=float,
        default=80.0,
        help="Max |t_bev - t_odom| and |t_bev - t_traj| in milliseconds",
    )
    p.add_argument("--max-arclen-m", type=float, default=40.0, help="Forward resampling along raceline (meters)")
    p.add_argument("--k-modes", type=int, default=4, help="K modes / model.num_heads (ignored label-wise for singleton)")
    p.add_argument(
        "--mode-label",
        type=str,
        default="kmeans",
        choices=("kmeans", "angular", "singleton"),
        help="How to assign mode_id: kmeans on traj endpoints (legacy), angular bins of chord (start→end), or all 0 (singleton; requires --k-modes 1)",
    )
    p.add_argument(
        "--traj-topic",
        type=str,
        default=DEFAULT_TRAJ_TOPIC,
        help="Polyline topic: /mpc/prediction (MarkerArray) or Autoware Trajectory topic (see planner_rosbag_recording.md)",
    )
    p.add_argument(
        "--traj-source",
        type=str,
        default="topic",
        choices=("topic", "odom_extrap"),
        help="topic: sync BEV+odom+traj-topic; odom_extrap: integrate twist from odom only (no third topic)",
    )
    p.add_argument("--val-ratio", type=float, default=0.12, help="Fraction of time-ordered samples for validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Remove existing *.npz in out dirs before writing")
    args = p.parse_args()

    bag = Path(args.bag)
    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    sync_slop_ns = int(args.sync_slop_ms * 1e6)
    rng = np.random.default_rng(args.seed)

    if args.overwrite:
        _clear_npz(out_train)
        _clear_npz(out_val)

    samples = extract_samples(
        bag,
        horizon=args.horizon,
        sync_slop_ns=sync_slop_ns,
        stride=args.stride,
        max_arclen_m=args.max_arclen_m,
        k_modes=args.k_modes,
        rng=rng,
        mode_scheme=args.mode_label,
        traj_topic=args.traj_topic,
        traj_source=args.traj_source,
    )
    nt, nv = write_npz_split(samples, out_train, out_val, val_ratio=args.val_ratio)
    print(f"Wrote {nt} train and {nv} val samples under {out_train} / {out_val}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
