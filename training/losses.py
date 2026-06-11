"""
Custom loss functions for trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """
    Combined loss: L1 + heading loss with optional uncertainty weighting.
    """

    def __init__(self, xy_weight: float = 1.0, heading_weight: float = 0.1, use_l1: bool = True):
        super().__init__()
        self.xy_weight = xy_weight
        self.heading_weight = heading_weight
        self.use_l1 = use_l1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, T, 3) - x, y, yaw
            target: (B, T, 3)
        """
        if self.use_l1:
            xy_loss = F.l1_loss(pred[:, :, :2], target[:, :, :2])
        else:
            xy_loss = F.mse_loss(pred[:, :, :2], target[:, :, :2])

        # Circular heading loss
        heading_diff = pred[:, :, 2] - target[:, :, 2]
        heading_loss = (1 - torch.cos(heading_diff)).mean()

        total_loss = self.xy_weight * xy_loss + self.heading_weight * heading_loss
        return total_loss, xy_loss, heading_loss
