"""Load per-sample .npz produced by prepare_data.py (or future bag extractors)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import C_BEV, H_BEV, W_BEV


class BevTrajectoryNpzDataset(Dataset):
    """Each file: bev (4,H,W), traj_gt (T,2), mode_id scalar, optional aux (D,)."""

    def __init__(
        self,
        data_dir: str | Path,
        aux_dim: int = 0,
        strict_shapes: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.aux_dim = aux_dim
        self.strict_shapes = strict_shapes
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir is not a directory: {self.data_dir}")
        self._files = sorted(self.data_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"No .npz files under {self.data_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def _load_npz(self, path: Path) -> tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
        z = np.load(path, allow_pickle=False)
        if "bev" not in z or "traj_gt" not in z or "mode_id" not in z:
            raise KeyError(f"{path}: expected keys bev, traj_gt, mode_id")
        bev = np.asarray(z["bev"], dtype=np.float32)
        traj = np.asarray(z["traj_gt"], dtype=np.float32)
        mid = int(np.asarray(z["mode_id"], dtype=np.int64).ravel()[0])
        aux = None
        if self.aux_dim > 0:
            if "aux" not in z:
                raise KeyError(f"{path}: aux_dim>0 but missing 'aux' array")
            aux = np.asarray(z["aux"], dtype=np.float32).reshape(-1)
            if aux.shape[0] != self.aux_dim:
                raise ValueError(f"{path}: aux shape {aux.shape} expected ({self.aux_dim},)")
        if self.strict_shapes:
            if bev.shape != (C_BEV, H_BEV, W_BEV):
                raise ValueError(f"{path}: bev shape {bev.shape} expected {(C_BEV, H_BEV, W_BEV)}")
            if traj.ndim != 2 or traj.shape[1] != 2:
                raise ValueError(f"{path}: traj_gt bad shape {traj.shape}")
        return bev, traj, mid, aux

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self._files[idx]
        bev, traj, mid, aux = self._load_npz(path)
        bev_t = torch.from_numpy(bev)
        traj_t = torch.from_numpy(traj)
        mid_t = torch.tensor(mid, dtype=torch.long)
        if self.aux_dim > 0:
            aux_t = torch.from_numpy(aux)  # type: ignore[arg-type]
            return bev_t, traj_t, mid_t, aux_t
        aux_empty = torch.zeros(self.aux_dim, dtype=torch.float32)
        return bev_t, traj_t, mid_t, aux_empty


def collate_with_optional_aux(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bevs, trajs, modes, auxs = zip(*batch)
    bev_b = torch.stack(bevs, dim=0)
    traj_b = torch.stack(trajs, dim=0)
    mode_b = torch.stack(modes, dim=0)
    aux_b = torch.stack(auxs, dim=0)
    return bev_b, traj_b, mode_b, aux_b
