#!/usr/bin/env python3
"""
Block-1 data audit: .npz train/val statistics + optional rosbag2 sync replay.

Run from planner_bev/:
  python3 scripts/audit_planner_npz.py --train-dir datasets/from_bag/train --val-dir datasets/from_bag/val
  python3 scripts/audit_planner_npz.py ... --json > audit.json

Optional bag timing (same gates as extract_data_from_bag.py):
  python3 scripts/audit_planner_npz.py --train-dir ... --val-dir ... \\
    --bag datasets/rosbag2_planner/planner_YYYYMMDD_HHMMSS

See design_docs/planner_npz_data_audit.md for interpretation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.extract_from_bag import DEFAULT_TRAJ_TOPIC, bag_sync_diagnostics  # noqa: E402


def _quantiles(a: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    if a.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in qs}


def _list_npz(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(d.glob("*.npz"))


def _load_one(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    z = np.load(path, allow_pickle=False)
    bev = np.asarray(z["bev"], dtype=np.float32)
    traj = np.asarray(z["traj_gt"], dtype=np.float32)
    mid = int(np.asarray(z["mode_id"]).ravel()[0])
    return bev, traj, mid


def _traj_stats(traj: np.ndarray) -> dict[str, float]:
    """Geometry-only proxies (ego frame: x forward, y left)."""
    x, y = traj[:, 0], traj[:, 1]
    dx = np.diff(x)
    mono_fwd = float(np.mean(dx >= -1e-3)) if dx.size else 0.0
    end_x, end_y = float(x[-1]), float(y[-1])
    lat_span = float(np.max(np.abs(y)))
    if traj.shape[0] >= 3:
        d2 = traj[2:] + traj[:-2] - 2.0 * traj[1:-1]
        curv = float(np.mean(d2**2))
    else:
        curv = 0.0
    return {
        "end_x": end_x,
        "end_y": end_y,
        "mono_fwd_step_ratio": mono_fwd,
        "lat_span_m": lat_span,
        "curv_proxy_mean_sq": curv,
    }


def _audit_split(name: str, paths: list[Path]) -> dict[str, Any]:
    if not paths:
        return {"name": name, "count": 0, "error": "no npz files"}

    modes: list[int] = []
    ends = []
    bev_mean_ch = []
    straight_flags = []

    mono: list[float] = []
    curv: list[float] = []
    for p in paths:
        bev, traj, mid = _load_one(p)
        modes.append(mid)
        ends.append(traj[-1].astype(np.float64))
        bev_mean_ch.append(bev.reshape(bev.shape[0], -1).mean(axis=1))
        st = _traj_stats(traj)
        mono.append(st["mono_fwd_step_ratio"])
        curv.append(st["curv_proxy_mean_sq"])
        straight = (st["end_x"] > 5.0) and (abs(st["end_y"]) < 2.5) and (st["lat_span_m"] < 4.0)
        straight_flags.append(bool(straight))

    ends_arr = np.stack(ends, axis=0)
    bev_mean = np.stack(bev_mean_ch, axis=0)
    mode_counts = Counter(modes)
    n = len(paths)

    out: dict[str, Any] = {
        "name": name,
        "count": n,
        "mode_id_counts": {str(k): mode_counts[k] for k in sorted(mode_counts)},
        "mode_id_fractions": {str(k): mode_counts[k] / n for k in sorted(mode_counts)},
        "endpoint_xy": {
            "mean": ends_arr.mean(axis=0).tolist(),
            "std": ends_arr.std(axis=0).tolist(),
            "quantiles_x": _quantiles(ends_arr[:, 0]),
            "quantiles_y": _quantiles(ends_arr[:, 1]),
        },
        "straight_heuristic_fraction": float(np.mean(straight_flags)),
        "mono_fwd_step_ratio_mean": float(np.mean(mono)),
        "curv_proxy_mean_sq_mean": float(np.mean(curv)),
        "bev_channel_mean_over_dataset": bev_mean.mean(axis=0).tolist(),
        "bev_channel_std_over_dataset": bev_mean.std(axis=0).tolist(),
    }
    return out


def _compare_splits(train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    if train.get("count", 0) == 0 or val.get("count", 0) == 0:
        return {"note": "skip compare: missing train or val"}

    # L2 distance between mean endpoints
    mt = np.asarray(train["endpoint_xy"]["mean"], dtype=np.float64)
    mv = np.asarray(val["endpoint_xy"]["mean"], dtype=np.float64)
    modes_t = set(train["mode_id_fractions"])
    modes_v = set(val["mode_id_fractions"])
    all_modes = sorted(modes_t | modes_v, key=lambda x: int(x))

    frac_diff = {}
    for k in all_modes:
        ft = float(train["mode_id_fractions"].get(k, 0.0))
        fv = float(val["mode_id_fractions"].get(k, 0.0))
        frac_diff[k] = round(fv - ft, 4)

    return {
        "mean_endpoint_l2_train_vs_val_m": float(np.linalg.norm(mt - mv)),
        "mode_id_fraction_delta_val_minus_train": frac_diff,
        "straight_heuristic_train": train.get("straight_heuristic_fraction"),
        "straight_heuristic_val": val.get("straight_heuristic_fraction"),
        "mono_fwd_step_ratio_mean_train": train.get("mono_fwd_step_ratio_mean"),
        "mono_fwd_step_ratio_mean_val": val.get("mono_fwd_step_ratio_mean"),
    }


def _print_report(
    train: dict[str, Any],
    val: dict[str, Any],
    compare: dict[str, Any],
    bag_diag: dict[str, Any] | None,
) -> None:
    def _p(d: dict[str, Any], title: str) -> None:
        print(f"\n=== {title} ===")
        if d.get("error"):
            print(f"  {d['error']}")
            return
        print(f"  count: {d['count']}")
        print(f"  mode_id counts: {d['mode_id_counts']}")
        print(f"  mode_id fractions: {d['mode_id_fractions']}")
        ex = d["endpoint_xy"]
        print(f"  endpoint mean (x,y) m: ({ex['mean'][0]:.3f}, {ex['mean'][1]:.3f})")
        print(f"  endpoint std  (x,y) m: ({ex['std'][0]:.3f}, {ex['std'][1]:.3f})")
        print(f"  endpoint x quantiles (m): {ex['quantiles_x']}")
        print(f"  endpoint y quantiles (m): {ex['quantiles_y']}")
        print(f"  straight_heuristic_fraction: {d['straight_heuristic_fraction']:.4f}")
        print(f"  mono_fwd_step_ratio_mean: {d['mono_fwd_step_ratio_mean']:.4f}")
        print(f"  curv_proxy_mean_sq_mean: {d['curv_proxy_mean_sq_mean']:.6f}")
        print(f"  bev channel mean (4ch): {[round(x, 4) for x in d['bev_channel_mean_over_dataset']]}")
        print(f"  bev channel std  (4ch): {[round(x, 4) for x in d['bev_channel_std_over_dataset']]}")

    _p(train, "TRAIN")
    _p(val, "VAL")
    print("\n=== TRAIN vs VAL (split drift) ===")
    for k, v in compare.items():
        print(f"  {k}: {v}")

    if bag_diag:
        print("\n=== BAG sync replay (extract_samples gates) ===")
        for k, v in bag_diag.items():
            if k.startswith("dt_") and isinstance(v, dict):
                print(f"  {k} ms: {v}")
            else:
                print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--val-dir", type=str, required=True)
    ap.add_argument("--json", action="store_true", help="Emit single JSON object to stdout")
    ap.add_argument("--bag", type=str, default=None, help="Optional rosbag2 dir for sync diagnostics")
    ap.add_argument("--sync-slop-ms", type=float, default=80.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--max-arclen-m", type=float, default=40.0)
    ap.add_argument(
        "--traj-topic",
        type=str,
        default=DEFAULT_TRAJ_TOPIC,
        help="Must match extract_data_from_bag.py when auditing sync",
    )
    ap.add_argument(
        "--traj-source",
        type=str,
        default="topic",
        choices=("topic", "odom_extrap"),
        help="Must match extract_data_from_bag.py",
    )
    args = ap.parse_args()

    train_dir = (_ROOT / args.train_dir).resolve()
    val_dir = (_ROOT / args.val_dir).resolve()

    train_paths = _list_npz(train_dir)
    val_paths = _list_npz(val_dir)

    train = _audit_split("train", train_paths)
    val = _audit_split("val", val_paths)
    compare = _compare_splits(train, val)

    bag_diag: dict[str, Any] | None = None
    if args.bag:
        bag_path = Path(args.bag).expanduser()
        if not bag_path.is_absolute():
            bag_path = (_ROOT / bag_path).resolve()
        sync_slop_ns = int(args.sync_slop_ms * 1e6)
        d = bag_sync_diagnostics(
            bag_path,
            sync_slop_ns=sync_slop_ns,
            stride=args.stride,
            horizon=args.horizon,
            max_arclen_m=args.max_arclen_m,
            traj_topic=args.traj_topic,
            traj_source=args.traj_source,
        )
        bag_diag = {
            "bag": str(bag_path),
            "traj_topic": args.traj_topic,
            "traj_source": args.traj_source,
            "n_bev_msgs": d.n_bev_msgs,
            "n_considered_after_stride": d.n_considered_after_stride,
            "skipped_odom_idx": d.skipped_odom_idx,
            "skipped_odom_slop": d.skipped_odom_slop,
            "skipped_traj_idx": d.skipped_traj_idx,
            "skipped_traj_slop": d.skipped_traj_slop,
            "skipped_bev_decode": d.skipped_bev_decode,
            "skipped_traj_polyline_short": d.skipped_traj_polyline_short,
            "n_synced": d.n_synced,
            "dt_bev_odom_ms": _quantiles(d.dt_bev_odom_ms),
            "dt_bev_traj_ms": _quantiles(d.dt_bev_traj_ms),
        }

    payload = {"train": train, "val": val, "train_val_compare": compare, "bag_sync": bag_diag}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_report(train, val, compare, bag_diag)


if __name__ == "__main__":
    main()
