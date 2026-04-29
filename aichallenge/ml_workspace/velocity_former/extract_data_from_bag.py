"""Extract trajectory and control data from ROS 2 bags for VelocityFormer training.

Reads `Trajectory` and `AckermannControlCommand` messages, synchronizes them by
nearest-neighbor timestamp, then writes per-bag `.npy` files into the output dir.

Output (per bag dir):
    trajectories.npy  shape: (N, point_num, 2)  trajectory (x, y) points sampled at fixed interval
    velocities.npy    shape: (N,)               longitudinal target velocity (m/s)
    steers.npy        shape: (N,)               lateral steering tire angle (rad)
"""

import argparse
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rosbags.highlevel import AnyReader

try:
    from rosbags.typesys import Stores, get_types_from_idl, get_typestore
except ImportError:  # rosbags<0.10
    Stores = None
    get_types_from_idl = None
    get_typestore = None


@dataclass
class ExtractionConfig:
    """Configuration parameters for data extraction."""

    control_topic: str
    trajectory_topic: str
    control_msg_type: str = "autoware_auto_control_msgs/msg/AckermannControlCommand"
    trajectory_msg_type: str = "autoware_auto_planning_msgs/msg/Trajectory"
    # 1 サンプルとして用いる trajectory のポイント数 (系列長)
    point_num: int = 12
    # ロードするポイントの間隔
    interval: int = 10
    # 有効サンプルとみなす最低ポイント数
    minimum_num: int = 150
    # 追加のIDLディレクトリ（autoware_auto_*など）。Noneのとき自動検出。
    idl_dirs: Optional[List[Path]] = field(default=None)


def build_typestore(idl_dirs: Optional[List[Path]]):
    """Build a rosbags typestore that knows about ROS2 Humble + custom IDLs."""
    if get_typestore is None or Stores is None:
        return None
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if not idl_dirs:
        return typestore
    for d in idl_dirs:
        for idl_path in d.rglob("*.idl"):
            try:
                text = idl_path.read_text()
                typestore.register(get_types_from_idl(text))
            except Exception:  # noqa: BLE001
                continue
    return typestore


def worker_init(debug_mode: bool) -> None:
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] [PID:%(process)d] %(message)s",
        force=True,
    )


def setup_logger(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] [PID:%(process)d] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def synchronize_data(src_times: np.ndarray, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor synchronization between two sorted time series."""
    if len(target_times) == 0:
        return np.array([]), np.array([])

    idx_sorted = np.searchsorted(target_times, src_times)
    idx_sorted = np.clip(idx_sorted, 0, len(target_times) - 1)
    prev_idx = np.clip(idx_sorted - 1, 0, len(target_times) - 1)

    time_diff_curr = np.abs(target_times[idx_sorted] - src_times)
    time_diff_prev = np.abs(target_times[prev_idx] - src_times)

    use_prev = time_diff_prev < time_diff_curr
    final_indices = np.where(use_prev, prev_idx, idx_sorted)
    final_deltas = np.where(use_prev, time_diff_prev, time_diff_curr)
    return final_indices, final_deltas


def trajectory_to_points(msg, config: ExtractionConfig) -> np.ndarray:
    """Convert a Trajectory message into a fixed-size (point_num, 2) (x, y) array.

    Returns an empty array when the trajectory is shorter than `minimum_num` points.
    """
    pts = msg.points
    n = len(pts)
    if n < config.minimum_num:
        return np.empty((0, 2), dtype=np.float32)

    sampled = np.empty((config.point_num, 2), dtype=np.float32)
    for k in range(config.point_num):
        idx = min(config.interval * k, n - 1)
        p = pts[idx].pose.position
        sampled[k, 0] = float(p.x)
        sampled[k, 1] = float(p.y)
    return sampled


def process_bag(bag_path: Path, output_root: Path, config: ExtractionConfig, debug: bool = False) -> None:
    logger = logging.getLogger(__name__)
    bag_name = bag_path.name
    out_dir = output_root / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()

    cmd_data: List[List[float]] = []
    cmd_times: List[int] = []
    traj_data: List[np.ndarray] = []
    traj_times: List[int] = []

    try:
        typestore = build_typestore(config.idl_dirs)
        reader_kwargs = {"default_typestore": typestore} if typestore is not None else {}
        with AnyReader([bag_path], **reader_kwargs) as reader:
            # bag が autoware の IDL を埋め込んでいる場合、AnyReader は default_typestore を
            # 無視して埋め込み定義を採用する。埋め込み IDL は std/geometry を include していない
            # ことがあるので、ROS2 Humble の標準型を後追いで backfill しておく。
            if typestore is not None:
                added = 0
                for tname, tdef in typestore.fielddefs.items():
                    if tname not in reader.typestore.fielddefs:
                        reader.typestore.fielddefs[tname] = tdef
                        if hasattr(typestore, "types") and tname in typestore.types:
                            reader.typestore.types[tname] = typestore.types[tname]
                        added += 1
                if added and hasattr(reader.typestore, "cache"):
                    reader.typestore.cache.clear()

            target_topics = [config.control_topic, config.trajectory_topic]
            connections = [c for c in reader.connections if c.topic in target_topics]
            if not connections:
                logger.warning(f"{bag_name}: No relevant topics found.")
                return

            for conn, timestamp, raw in reader.messages(connections=connections):
                try:
                    msg = reader.deserialize(raw, conn.msgtype)

                    if conn.topic == config.control_topic:
                        if conn.msgtype == config.control_msg_type:
                            vel = msg.longitudinal.speed
                            steer = msg.lateral.steering_tire_angle
                            cmd_data.append([float(vel), float(steer)])
                            cmd_times.append(timestamp)

                    elif conn.topic == config.trajectory_topic:
                        if conn.msgtype == config.trajectory_msg_type:
                            sampled = trajectory_to_points(msg, config)
                            if sampled.size == 0:
                                continue
                            traj_data.append(sampled)
                            traj_times.append(timestamp)
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"Failed to read {bag_name}: {e}")
        return

    if not cmd_data or not traj_data:
        logger.warning(f"Skipping {bag_name}: insufficient data (cmd={len(cmd_data)}, traj={len(traj_data)}).")
        return

    np_cmd = np.array(cmd_data, dtype=np.float32)
    np_cmd_times = np.array(cmd_times, dtype=np.int64)
    np_traj = np.stack(traj_data).astype(np.float32)
    np_traj_times = np.array(traj_times, dtype=np.int64)

    sort_idx = np.argsort(np_cmd_times)
    np_cmd_times = np_cmd_times[sort_idx]
    np_cmd = np_cmd[sort_idx]

    indices, deltas = synchronize_data(np_traj_times, np_cmd_times)
    synced_cmds = np_cmd[indices]
    synced_velocities = synced_cmds[:, 0]
    synced_steers = synced_cmds[:, 1]

    np.save(out_dir / "trajectories.npy", np_traj)
    np.save(out_dir / "velocities.npy", synced_velocities)
    np.save(out_dir / "steers.npy", synced_steers)
    if debug:
        np.save(out_dir / "delta_times.npy", deltas / 1e9)

    duration = time.perf_counter() - t_total_start
    logger.info(f"Saved {bag_name}: {len(np_traj)} samples ({duration:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trajectory and control data from ROS 2 bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bags-dir", type=Path, help="Directory containing rosbag folders (recursive).")
    group.add_argument("--seq-dirs", type=Path, nargs="+", help="List of sequence directories to process.")
    parser.add_argument("--outdir", type=Path, required=True, help="Root directory for output files.")

    parser.add_argument(
        "--control-topic",
        type=str,
        default="/control/command/control_cmd",
        help="Topic for AckermannControlCommand.",
    )
    parser.add_argument(
        "--trajectory-topic",
        type=str,
        default="/planning/scenario_planning/trajectory",
        help="Topic for Trajectory.",
    )
    parser.add_argument("--point-num", type=int, default=12, help="Number of trajectory points per sample.")
    parser.add_argument("--interval", type=int, default=10, help="Index step between sampled trajectory points.")
    parser.add_argument("--minimum-num", type=int, default=150, help="Minimum trajectory length to be valid.")
    parser.add_argument(
        "--idl-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing custom IDL files (e.g. autoware_auto_*). May be specified multiple times.",
    )

    default_workers = min(os.cpu_count() or 1, 8)
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of parallel workers.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed logging.")

    args = parser.parse_args()
    setup_logger(args.debug)
    logger = logging.getLogger(__name__)

    bag_dirs: List[Path] = []
    if args.bags_dir:
        p = args.bags_dir.expanduser().resolve()
        bag_dirs = [x.parent for x in p.rglob("metadata.yaml")]
        if not bag_dirs and (p / "metadata.yaml").exists():
            bag_dirs = [p]
    elif args.seq_dirs:
        for p in args.seq_dirs:
            p = p.expanduser().resolve()
            if (p / "metadata.yaml").exists():
                bag_dirs.append(p)

    bag_dirs = sorted(set(bag_dirs))
    if not bag_dirs:
        logger.error("No valid ROS 2 bag directories found.")
        return

    num_workers = min(max(1, args.workers), len(bag_dirs))
    logger.info(f"Found {len(bag_dirs)} bags. Starting with {num_workers} workers.")

    config = ExtractionConfig(
        control_topic=args.control_topic,
        trajectory_topic=args.trajectory_topic,
        point_num=args.point_num,
        interval=args.interval,
        minimum_num=args.minimum_num,
        idl_dirs=args.idl_dir,
    )
    tasks = [(p, args.outdir, config, args.debug) for p in bag_dirs]

    start = time.time()
    with multiprocessing.Pool(
        processes=num_workers, initializer=worker_init, initargs=(args.debug,)
    ) as pool:
        pool.starmap(process_bag, tasks)
    logger.info(f"All processing finished in {time.time() - start:.2f}s.")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
