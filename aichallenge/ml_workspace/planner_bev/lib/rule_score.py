"""Rule-based trajectory scoring (numpy) — aligns with design_docs §3.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BEVGridSpec:
    """Same convention as bev_scene_stack (ego base_link, x forward, y left)."""

    x_min: float = -12.0
    x_max: float = 52.0
    y_min: float = -18.0
    y_max: float = 18.0
    res: float = 0.25
    h: int = 256
    w: int = 144

    def xy_to_rc(self, x: float, y: float) -> Tuple[int, int]:
        row = int((self.x_max - x) / self.res)
        col = int((self.y_max - y) / self.res)
        return row, col


def score_trajectory(
    bev: np.ndarray,
    traj_xy: np.ndarray,
    spec: BEVGridSpec | None = None,
    w_obs: float = 1.0,
    w_lane: float = 0.3,
    w_curv: float = 0.05,
) -> float:
    """
    Lower is better. Uses channel 2=obstacles, 0=lane (higher=on-lane good -> negate as cost).

    Args:
        bev: (4, H, W) float
        traj_xy: (T, 2) ego-frame meters
    """
    if spec is None:
        spec = BEVGridSpec()
    ch_lane = bev[0]
    ch_obs = bev[2]
    t = traj_xy.shape[0]
    c_obs = 0.0
    c_lane = 0.0
    for ti in range(t):
        x, y = float(traj_xy[ti, 0]), float(traj_xy[ti, 1])
        r, c = spec.xy_to_rc(x, y)
        if 0 <= r < spec.h and 0 <= c < spec.w:
            c_obs += float(ch_obs[r, c])
            # prefer high lane occupancy under footprint
            c_lane += max(0.0, 0.5 - float(ch_lane[r, c]))
    c_obs /= max(t, 1)
    c_lane /= max(t, 1)
    if t >= 3:
        d2 = traj_xy[2:] + traj_xy[:-2] - 2.0 * traj_xy[1:-1]
        c_curv = float(np.mean(d2**2))
    else:
        c_curv = 0.0
    return w_obs * c_obs + w_lane * c_lane + w_curv * c_curv


def select_best_trajectory(
    bev: np.ndarray,
    trajs: np.ndarray,
    spec: BEVGridSpec | None = None,
    **kwargs: float,
) -> int:
    """
    Args:
        trajs: (K, T, 2)
    Returns:
        index of minimum score
    """
    scores = [score_trajectory(bev, trajs[k], spec=spec, **kwargs) for k in range(trajs.shape[0])]
    return int(np.argmin(scores))
