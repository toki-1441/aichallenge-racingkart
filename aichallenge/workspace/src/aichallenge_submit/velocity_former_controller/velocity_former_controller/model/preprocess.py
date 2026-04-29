"""Trajectory preprocessing for VelocityFormer inference.

Mirrors `lib/data.trajectory_to_token_ids` from the ml_workspace, but expressed
purely in NumPy with no PyTorch dependency.
"""

import numpy as np


def trajectory_to_token_ids(traj_xy: np.ndarray) -> np.ndarray:
    """Convert a (P, 2) trajectory into (P,) integer-degree token IDs.

    Args:
        traj_xy: Trajectory points, shape (P, 2). The first axis is sequence,
            second is (x, y).

    Returns:
        np.ndarray of shape (P,) and dtype int64 with values in [0, 360).
    """
    n = traj_xy.shape[0]
    if n < 2:
        return np.zeros((n,), dtype=np.int64)

    diffs = np.empty_like(traj_xy)
    diffs[1:] = traj_xy[1:] - traj_xy[:-1]
    diffs[0] = diffs[1]

    angles_rad = np.arctan2(diffs[:, 1], diffs[:, 0])
    angles_deg = np.rad2deg(angles_rad)
    angles_deg = np.mod(angles_deg, 360.0)
    return (np.round(angles_deg).astype(np.int64) % 360)


def sample_trajectory_points(
    traj_xy: np.ndarray, point_num: int, interval: int
) -> np.ndarray:
    """Sample `point_num` (x, y) points from `traj_xy` at fixed `interval`.

    If the trajectory is shorter than required, the last point is repeated.

    Args:
        traj_xy: Trajectory points, shape (M, 2).
        point_num: Output sequence length.
        interval: Index spacing between sampled points.

    Returns:
        np.ndarray of shape (point_num, 2), float32.
    """
    m = traj_xy.shape[0]
    sampled = np.empty((point_num, 2), dtype=np.float32)
    for k in range(point_num):
        idx = min(interval * k, max(0, m - 1))
        sampled[k] = traj_xy[idx]
    return sampled
