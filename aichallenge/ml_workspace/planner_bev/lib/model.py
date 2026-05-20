"""CNN encoder + K trajectory heads (ego-frame (x,y) over T steps)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import C_BEV


def _conv_out_size(
    size: int, kernel: int, stride: int, padding: int, dilation: int = 1
) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class BevTrajectoryNet(nn.Module):
    """Encode BEV (B,4,H,W) (+ optional aux) and predict K trajectories (B,K,T,2)."""

    def __init__(
        self,
        h_bev: int = 256,
        w_bev: int = 144,
        c_in: int = C_BEV,
        aux_dim: int = 0,
        num_heads: int = 4,
        horizon: int = 40,
        stem_channels: Tuple[int, ...] = (32, 64, 128, 256),
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h_bev = h_bev
        self.w_bev = w_bev
        self.c_in = c_in
        self.aux_dim = aux_dim
        self.num_heads = num_heads
        self.horizon = horizon

        layers: list[nn.Module] = []
        in_ch = c_in
        h, w = h_bev, w_bev
        for out_ch in stem_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            h = _conv_out_size(h, 3, 2, 1)
            w = _conv_out_size(w, 3, 2, 1)
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.encoder_out_ch = in_ch

        fused_in = self.encoder_out_ch + aux_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(embed_dim, horizon * 2) for _ in range(num_heads)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bev: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bev: (B, C, H, W)
            aux: (B, aux_dim) or None
        Returns:
            (B, K, T, 2)
        """
        if bev.dim() != 4:
            raise ValueError(f"bev must be (B,C,H,W), got {bev.shape}")
        b = bev.shape[0]
        x = self.encoder(bev)
        x = self.gap(x).flatten(1)  # (B, encoder_out_ch)
        if self.aux_dim > 0:
            if aux is None:
                aux = bev.new_zeros(b, self.aux_dim)
            if aux.shape != (b, self.aux_dim):
                raise ValueError(f"aux must be (B,{self.aux_dim}), got {aux.shape}")
            x = torch.cat([x, aux], dim=1)
        elif aux is not None:
            raise ValueError("aux passed but model.aux_dim == 0")
        z = self.fuse(x)
        outs = []
        for h in self.heads:
            o = h(z).view(b, self.horizon, 2)
            outs.append(o)
        return torch.stack(outs, dim=1)
