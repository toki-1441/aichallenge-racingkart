"""Offline rosbag2 → (bev, traj_gt, mode_id) .npz samples for planner_bev."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rosbags.highlevel import AnyReader

from .schema import C_BEV, H_BEV, W_BEV

TOPIC_BEV = "/bev_scene_stack/tensor"
TOPIC_ODOM = "/localization/kinematic_state"
# MPC publishes dynamic prediction at control rate (see multi_purpose_mpc_ros mpc_controller.py).
DEFAULT_TRAJ_TOPIC = "/mpc/prediction"
# Legacy static CSV trajectory (1 Hz); use --traj-topic for old bags.
LEGACY_TRAJ_TOPIC = "/planning/scenario_planning/trajectory"

# visualization_msgs/Marker type constants (ROS 2)
_MARKER_ADD = 0
_MARKER_SPHERE = 2
_MARKER_LINE_STRIP = 4


@dataclass
class StampMsg:
    stamp_ns: int
    msg: Any


def _quat_to_rot2d(q: Any) -> np.ndarray:
    """Rotation (2,2): v_map = R @ v_base (planar yaw from quaternion)."""
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def map_points_to_base(xy_m: np.ndarray, t_m: np.ndarray, R_mb: np.ndarray) -> np.ndarray:
    """xy_m: (N,2) in map, t_m ego origin in map, R_mb: v_map = R_mb @ v_base."""
    d = xy_m - t_m.reshape(1, 2)
    return (R_mb.T @ d.T).T.astype(np.float32)


def polyline_forward_resample(
    xy: np.ndarray,
    ego_xy: np.ndarray,
    num_points: int,
    max_arclen_m: float = 40.0,
) -> np.ndarray:
    """From polyline xy (N,2) in map, walk forward from closest vertex to ego."""
    if xy.shape[0] < 2:
        return np.tile(ego_xy.astype(np.float32), (num_points, 1))

    d2 = np.sum((xy - ego_xy.reshape(1, 2)) ** 2, axis=1)
    i0 = int(np.argmin(d2))
    sub = xy[i0:]
    if sub.shape[0] < 2:
        return np.tile(sub[0].astype(np.float32), (num_points, 1))

    seg = np.linalg.norm(np.diff(sub, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    L = min(max_arclen_m, total)
    if L <= 1e-6:
        return np.tile(sub[0].astype(np.float32), (num_points, 1))

    targets = np.linspace(0.0, L, num_points, dtype=np.float64)
    out = np.zeros((num_points, 2), dtype=np.float32)
    for j, s in enumerate(targets):
        idx = int(bisect.bisect_right(cum, s) - 1)
        idx = max(0, min(idx, sub.shape[0] - 2))
        denom = cum[idx + 1] - cum[idx]
        u = 0.0 if denom < 1e-9 else (s - cum[idx]) / denom
        u = float(np.clip(u, 0.0, 1.0))
        out[j] = ((1.0 - u) * sub[idx] + u * sub[idx + 1]).astype(np.float32)
    return out


def bev_from_float32_multiarray(msg: Any) -> np.ndarray:
    data = np.asarray(msg.data, dtype=np.float32)
    expected = C_BEV * H_BEV * W_BEV
    if data.size != expected:
        raise ValueError(f"BEV flat size {data.size} != {expected}")
    return data.reshape(C_BEV, H_BEV, W_BEV)


def odom_map_pose(msg: Any) -> tuple[np.ndarray, np.ndarray]:
    p = msg.pose.pose.position
    t = np.array([float(p.x), float(p.y)], dtype=np.float64)
    R = _quat_to_rot2d(msg.pose.pose.orientation)
    return t, R


def trajectory_xy_map(msg: Any) -> np.ndarray:
    pts = []
    for p in msg.points:
        pts.append([float(p.pose.position.x), float(p.pose.position.y)])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def marker_array_xy_map(msg: Any) -> np.ndarray:
    """
    Map-frame polyline from visualization_msgs/MarkerArray.
    Supports LINE_STRIP (first marker with >=2 points) or multiple SPHERE markers (sorted by id),
    as produced by mpc_controller._publish_mpc_pred_marker.
    """
    if not hasattr(msg, "markers") or not msg.markers:
        return np.zeros((0, 2), dtype=np.float64)
    active = [m for m in msg.markers if int(m.action) == _MARKER_ADD]
    if not active:
        return np.zeros((0, 2), dtype=np.float64)
    for m in active:
        if int(m.type) == _MARKER_LINE_STRIP and len(m.points) >= 2:
            return np.asarray([[float(p.x), float(p.y)] for p in m.points], dtype=np.float64)
    spheres = [m for m in active if int(m.type) == _MARKER_SPHERE]
    if len(spheres) >= 2:
        spheres.sort(key=lambda m: int(m.id))
        return np.asarray(
            [[float(m.pose.position.x), float(m.pose.position.y)] for m in spheres],
            dtype=np.float64,
        )
    return np.zeros((0, 2), dtype=np.float64)


def polyline_xy_map_from_msg(msg: Any) -> np.ndarray:
    """Autoware Trajectory or visualization MarkerArray → (N,2) in map frame."""
    if hasattr(msg, "markers"):
        return marker_array_xy_map(msg)
    if hasattr(msg, "points"):
        return trajectory_xy_map(msg)
    return np.zeros((0, 2), dtype=np.float64)


def odom_extrap_polyline_map(odom_msg: Any, *, num_points: int = 160, dt: float = 0.05) -> np.ndarray:
    """
    Dense forward polyline in map frame from a single Odometry sample (twist in base_link).
    Used when no dynamic trajectory topic is available (fallback).
    """
    t0, _R0 = odom_map_pose(odom_msg)
    q = odom_msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)), 1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2))
    vx = float(odom_msg.twist.twist.linear.x)
    vy = float(odom_msg.twist.twist.linear.y)
    wz = float(odom_msg.twist.twist.angular.z)
    x = float(t0[0])
    y = float(t0[1])
    out = np.zeros((num_points, 2), dtype=np.float64)
    for i in range(num_points):
        out[i, 0] = x
        out[i, 1] = y
        c, s = math.cos(yaw), math.sin(yaw)
        x += dt * (vx * c - vy * s)
        y += dt * (vx * s + vy * c)
        yaw += dt * wz
    return out


def _kmeans_labels(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(k, n))
    idx = rng.choice(n, size=k, replace=False)
    centers = x[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = dist.argmin(axis=1)
        new_c = centers.copy()
        for ki in range(k):
            mask = labels == ki
            if np.any(mask):
                new_c[ki] = x[mask].mean(axis=0)
        if np.allclose(new_c, centers):
            break
        centers = new_c
    return labels


def _angular_bin_labels(trajs: list[np.ndarray], k: int) -> np.ndarray:
    """
    Label by azimuth of chord (end - start) in ego base frame, K equal bins on [-pi, pi).
    More stable than endpoint k-means when many trajectories share similar endpoints (e.g. straight).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    n = len(trajs)
    out = np.zeros(n, dtype=np.int64)
    width = 2.0 * math.pi / float(k)
    for i, tr in enumerate(trajs):
        d = tr[-1].astype(np.float64) - tr[0].astype(np.float64)
        ang = math.atan2(float(d[1]), float(d[0]))
        a = (ang + math.pi) % (2.0 * math.pi)
        b = int(a // width) if width > 1e-9 else 0
        out[i] = min(max(b, 0), k - 1)
    return out


def required_topics_present(
    reader: AnyReader, traj_topic: str, traj_source: str = "topic"
) -> tuple[bool, list[str], dict[str, int]]:
    tops = reader.topics
    need = [TOPIC_BEV, TOPIC_ODOM]
    if traj_source != "odom_extrap":
        need.append(traj_topic)
    counts = {t: tops[t].msgcount if t in tops else 0 for t in need}
    ok = all(counts[t] > 0 for t in need)
    missing = [t for t in need if counts[t] == 0]
    return ok, missing, counts


def _collect_messages(
    reader: AnyReader, traj_topic: str, traj_source: str = "topic"
) -> tuple[list[StampMsg], list[StampMsg], list[StampMsg]]:
    conn_by_topic = {c.topic: c for c in reader.connections}
    topics: list[str] = [TOPIC_BEV, TOPIC_ODOM]
    if traj_source != "odom_extrap":
        topics.append(traj_topic)
    for t in topics:
        if t not in conn_by_topic:
            raise RuntimeError(f"Topic {t} not in bag connections")
    conns = [conn_by_topic[t] for t in topics]
    bev_l: list[StampMsg] = []
    odom_l: list[StampMsg] = []
    traj_l: list[StampMsg] = []
    for conn, ts, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        if conn.topic == TOPIC_BEV:
            bev_l.append(StampMsg(ts, msg))
        elif conn.topic == TOPIC_ODOM:
            odom_l.append(StampMsg(ts, msg))
        elif traj_source != "odom_extrap" and conn.topic == traj_topic:
            traj_l.append(StampMsg(ts, msg))
    bev_l.sort(key=lambda x: x.stamp_ns)
    odom_l.sort(key=lambda x: x.stamp_ns)
    traj_l.sort(key=lambda x: x.stamp_ns)
    return bev_l, odom_l, traj_l


@dataclass
class BagSyncDiagnostics:
    """Counts and timing deltas for matched (BEV, odom, trajectory) frames (same gates as extract_samples)."""

    n_bev_msgs: int
    n_considered_after_stride: int
    skipped_odom_idx: int
    skipped_odom_slop: int
    skipped_traj_idx: int
    skipped_traj_slop: int
    skipped_bev_decode: int
    skipped_traj_polyline_short: int
    n_synced: int
    dt_bev_odom_ms: np.ndarray
    dt_bev_traj_ms: np.ndarray


def bag_sync_diagnostics(
    bag_path: Path,
    *,
    sync_slop_ns: int,
    stride: int,
    horizon: int,
    max_arclen_m: float,
    traj_topic: str = DEFAULT_TRAJ_TOPIC,
    traj_source: str = "topic",
) -> BagSyncDiagnostics:
    """
    Replay the synchronization gates from extract_samples without k-means / .npz.
    dt_* are (t_bev - t_nearest) in milliseconds, non-negative for accepted rows.
    traj_source odom_extrap: no third-topic sync; dt_bev_traj_ms is zeros.
    """
    bag_path = bag_path.expanduser().resolve()
    if not bag_path.is_dir():
        raise FileNotFoundError(f"Not a rosbag2 directory: {bag_path}")

    with AnyReader([bag_path]) as reader:
        ok, missing, counts = required_topics_present(reader, traj_topic, traj_source)
        if not ok:
            raise RuntimeError(
                "Bag is missing required topics with messages: "
                f"{missing}. Present counts: {counts}."
            )
        bev_l, odom_l, traj_l = _collect_messages(reader, traj_topic, traj_source)

    odom_ts = [x.stamp_ns for x in odom_l]
    traj_ts = [x.stamp_ns for x in traj_l] if traj_source != "odom_extrap" else []

    skipped_odom_idx = skipped_odom_slop = 0
    skipped_traj_idx = skipped_traj_slop = 0
    skipped_bev_decode = skipped_traj_polyline_short = 0
    dt_odom: list[float] = []
    dt_traj: list[float] = []
    n_considered = 0

    for bi, bm in enumerate(bev_l):
        if stride > 1 and bi % stride != 0:
            continue
        n_considered += 1
        ts = bm.stamp_ns
        io = bisect.bisect_right(odom_ts, ts) - 1
        if io < 0:
            skipped_odom_idx += 1
            continue
        d_odom = ts - odom_ts[io]
        if d_odom > sync_slop_ns:
            skipped_odom_slop += 1
            continue

        d_traj = 0.0
        if traj_source != "odom_extrap":
            it = bisect.bisect_right(traj_ts, ts + sync_slop_ns) - 1
            if it < 0:
                skipped_traj_idx += 1
                continue
            d_traj = ts - traj_ts[it]
            if d_traj > sync_slop_ns:
                skipped_traj_slop += 1
                continue

        try:
            _bev = bev_from_float32_multiarray(bm.msg)
        except ValueError:
            skipped_bev_decode += 1
            continue

        odom = odom_l[io].msg
        if traj_source == "odom_extrap":
            xy_map = odom_extrap_polyline_map(odom)
        else:
            traj_msg = traj_l[it].msg
            xy_map = polyline_xy_map_from_msg(traj_msg)
        if xy_map.shape[0] < 2:
            skipped_traj_polyline_short += 1
            continue

        t_m, R_mb = odom_map_pose(odom)
        ego_xy = t_m
        xy_fwd = polyline_forward_resample(xy_map, ego_xy, horizon, max_arclen_m=max_arclen_m)
        _ = map_points_to_base(xy_fwd.astype(np.float64), t_m, R_mb)

        dt_odom.append(d_odom / 1e6)
        dt_traj.append(d_traj / 1e6)

    return BagSyncDiagnostics(
        n_bev_msgs=len(bev_l),
        n_considered_after_stride=n_considered,
        skipped_odom_idx=skipped_odom_idx,
        skipped_odom_slop=skipped_odom_slop,
        skipped_traj_idx=skipped_traj_idx,
        skipped_traj_slop=skipped_traj_slop,
        skipped_bev_decode=skipped_bev_decode,
        skipped_traj_polyline_short=skipped_traj_polyline_short,
        n_synced=len(dt_odom),
        dt_bev_odom_ms=np.asarray(dt_odom, dtype=np.float64),
        dt_bev_traj_ms=np.asarray(dt_traj, dtype=np.float64),
    )


def extract_samples(
    bag_path: Path,
    horizon: int,
    sync_slop_ns: int,
    stride: int,
    max_arclen_m: float,
    k_modes: int,
    rng: np.random.Generator,
    mode_scheme: str = "kmeans",
    traj_topic: str = DEFAULT_TRAJ_TOPIC,
    traj_source: str = "topic",
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Returns list of (bev, traj_gt, mode_id, stamp_ns).

    traj_topic: e.g. /mpc/prediction (MarkerArray) or legacy Autoware Trajectory topic.
    traj_source:
      - "topic": sync BEV + odom + traj_topic (Trajectory or MarkerArray).
      - "odom_extrap": no third topic; build a map-frame polyline by integrating twist from odom (fallback).

    mode_scheme:
      - "kmeans": cluster trajectory endpoints in R^2 (legacy default).
      - "angular": bin chord direction (start→end) into K equal azimuth sectors.
      - "singleton": all mode_id=0 (use k_modes=1 and model.num_heads=1).
    """
    bag_path = bag_path.expanduser().resolve()
    if not bag_path.is_dir():
        raise FileNotFoundError(f"Not a rosbag2 directory: {bag_path}")

    with AnyReader([bag_path]) as reader:
        ok, missing, counts = required_topics_present(reader, traj_topic, traj_source)
        if not ok:
            raise RuntimeError(
                "Bag is missing required topics with messages: "
                f"{missing}. Present counts: {counts}. "
                "Record /mpc/prediction (MPC) or pass --traj-topic / --traj-source odom_extrap. "
                "See design_docs/planner_rosbag_recording.md."
            )

        bev_l, odom_l, traj_l = _collect_messages(reader, traj_topic, traj_source)

    odom_ts = [x.stamp_ns for x in odom_l]
    traj_ts = [x.stamp_ns for x in traj_l] if traj_source != "odom_extrap" else []

    raw_samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    for bi, bm in enumerate(bev_l):
        if stride > 1 and bi % stride != 0:
            continue
        ts = bm.stamp_ns
        io = bisect.bisect_right(odom_ts, ts) - 1
        if io < 0:
            continue
        if ts - odom_ts[io] > sync_slop_ns:
            continue

        if traj_source != "odom_extrap":
            it = bisect.bisect_right(traj_ts, ts + sync_slop_ns) - 1
            if it < 0:
                continue
            if ts - traj_ts[it] > sync_slop_ns:
                continue

        try:
            bev = bev_from_float32_multiarray(bm.msg)
        except ValueError:
            continue

        odom = odom_l[io].msg
        if traj_source == "odom_extrap":
            xy_map = odom_extrap_polyline_map(odom)
        else:
            traj_msg = traj_l[it].msg
            xy_map = polyline_xy_map_from_msg(traj_msg)
        if xy_map.shape[0] < 2:
            continue

        t_m, R_mb = odom_map_pose(odom)
        ego_xy = t_m
        xy_fwd = polyline_forward_resample(xy_map, ego_xy, horizon, max_arclen_m=max_arclen_m)
        traj_base = map_points_to_base(xy_fwd.astype(np.float64), t_m, R_mb)
        raw_samples.append((bev, traj_base, ts))

    if not raw_samples:
        raise RuntimeError(
            "No synchronized (BEV, odom, trajectory) samples produced. "
            "Check timestamps, stride, traj_topic / traj_source, and that the trajectory source yields >=2 map points."
        )

    trajs_only = [s[1] for s in raw_samples]
    scheme = mode_scheme.strip().lower()
    if scheme == "singleton":
        if k_modes != 1:
            raise ValueError("mode_scheme=singleton requires k_modes=1 (and model.num_heads=1 in training).")
        labels = np.zeros(len(raw_samples), dtype=np.int64)
    elif scheme == "angular":
        labels = _angular_bin_labels(trajs_only, k_modes)
    elif scheme == "kmeans":
        ends = np.stack([s[1][-1].astype(np.float64) for s in raw_samples], axis=0)
        labels = _kmeans_labels(ends, k_modes, rng)
    else:
        raise ValueError(f"Unknown mode_scheme: {mode_scheme!r} (use kmeans, angular, singleton)")
    out: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for i, s in enumerate(raw_samples):
        bev, traj, st = s
        out.append((bev, traj, int(labels[i]), int(st)))
    return out


def write_npz_split(
    samples: Sequence[tuple[np.ndarray, np.ndarray, int, int]],
    out_train: Path,
    out_val: Path,
    val_ratio: float,
) -> tuple[int, int]:
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(samples)), key=lambda i: samples[i][3])
    n = len(order)
    n_val = max(1, int(round(n * val_ratio))) if n >= 2 else 0
    if n < 2:
        n_val = 0
    val_set = set(order[-n_val:]) if n_val else set()
    nt = nv = 0
    for j, idx in enumerate(order):
        bev, traj, mid, _st = samples[idx]
        dest = out_val if idx in val_set else out_train
        name = f"{j:06d}.npz"
        np.savez_compressed(dest / name, bev=bev, traj_gt=traj, mode_id=np.int64(mid))
        if idx in val_set:
            nv += 1
        else:
            nt += 1
    return nt, nv
