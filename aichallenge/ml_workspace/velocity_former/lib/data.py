"""Dataset utilities for VelocityFormer training.

Loads `trajectories.npy`, `velocities.npy`, `steers.npy` produced by
`extract_data_from_bag.py`, converts trajectories to integer-degree token
sequences, and yields `(input_ids, label)` pairs.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)


def trajectory_to_token_ids(traj_xy: np.ndarray) -> np.ndarray:
    """Convert a (point_num, 2) trajectory into a (point_num,) integer token sequence.

    Each token encodes the angle (in whole degrees, 0..359) between consecutive
    points relative to the +x axis. The first point's "previous direction" is
    defined to be the same as the second point's, so the output length matches
    the input length.
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
    return np.round(angles_deg).astype(np.int64) % 360


class TrajectoryControlSequenceDataset(Dataset):
    """Loads trajectory/control samples from a single sequence directory."""

    def __init__(
        self,
        seq_dir: Union[str, Path],
        label_type: str = "velocity",
        max_steering: float = 0.7,
        min_steering: float = -0.7,
        flip_prob: float = 0.0,
    ):
        self.seq_dir = Path(seq_dir)
        self.label_type = label_type
        self.max_steering = max_steering
        self.min_steering = min_steering
        self.flip_prob = flip_prob

        try:
            self.trajectories = np.load(self.seq_dir / "trajectories.npy")  # (N, P, 2)
            self.velocities = np.load(self.seq_dir / "velocities.npy")  # (N,)
            self.steers = np.load(self.seq_dir / "steers.npy")  # (N,)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.seq_dir}: {e}")

        n = len(self.trajectories)
        if not (len(self.velocities) == n and len(self.steers) == n):
            raise ValueError(
                f"Data length mismatch in {self.seq_dir}: "
                f"traj={n}, vel={len(self.velocities)}, steer={len(self.steers)}"
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    def _augment(self, traj: np.ndarray, label: float) -> Tuple[np.ndarray, float]:
        """Random mirror augmentation. Mirrors trajectory in y, flips steering sign."""
        if self.flip_prob <= 0.0:
            return traj, label
        if np.random.rand() < self.flip_prob:
            traj = traj.copy()
            traj[:, 1] = -traj[:, 1]
            if self.label_type == "steering":
                label = -label
        return traj, label

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        traj = self.trajectories[idx].astype(np.float32)

        if self.label_type == "velocity":
            label = float(self.velocities[idx])
        elif self.label_type == "steering":
            label = float(np.clip(self.steers[idx], self.min_steering, self.max_steering))
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        traj, label = self._augment(traj, label)
        token_ids = trajectory_to_token_ids(traj)
        target = np.array([label], dtype=np.float32)
        return token_ids, target


class MultiSeqConcatDataset(ConcatDataset):
    """Aggregates multiple sequence directories under a single root."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        label_type: str = "velocity",
        max_steering: float = 0.7,
        min_steering: float = -0.7,
        flip_prob: float = 0.0,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        dataset_root = Path(dataset_root)
        all_seq_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

        target_seq_dirs: List[Path] = []
        for p in all_seq_dirs:
            name = p.name
            if include and not any(inc in name for inc in include):
                continue
            if exclude and any(exc in name for exc in exclude):
                continue
            target_seq_dirs.append(p)

        datasets: List[Dataset] = []
        required = ["trajectories.npy", "velocities.npy", "steers.npy"]
        for seq_dir in target_seq_dirs:
            if not all((seq_dir / f).exists() for f in required):
                logger.warning(f"Skipping {seq_dir.name}: Missing .npy files.")
                continue
            try:
                ds = TrajectoryControlSequenceDataset(
                    seq_dir,
                    label_type=label_type,
                    max_steering=max_steering,
                    min_steering=min_steering,
                    flip_prob=flip_prob,
                )
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to load sequence {seq_dir}: {e}")

        if not datasets:
            raise RuntimeError(f"No valid sequences found in {dataset_root}.")

        super().__init__(datasets)
        logger.info(f"Loaded {len(datasets)} sequences from {dataset_root}. Total samples: {len(self)}")
