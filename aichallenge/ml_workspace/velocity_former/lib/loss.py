"""Loss functions for VelocityFormer training."""

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """SmoothL1 / Huber loss with configurable beta.

    The output is a scalar. Equivalent to `torch.nn.SmoothL1Loss(beta=beta)`,
    wrapped here for parity with the tiny_lidar_net workspace style.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(beta=beta)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets)
