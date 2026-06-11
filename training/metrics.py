"""
Evaluation metrics for trajectory prediction.
ADE: Average Displacement Error
FDE: Final Displacement Error
MR: Miss Rate
"""

import torch
import numpy as np
from typing import Tuple


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error over all timesteps.
    Args:
        pred: (B, T, 2) predicted x, y
        target: (B, T, 2) ground truth x, y
    Returns:
        ade: scalar
    """
    displacement = torch.norm(pred[:, :, :2] - target[:, :, :2], dim=-1)  # (B, T)
    ade = displacement.mean()
    return ade


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error at last timestep.
    Args:
        pred: (B, T, 2) predicted x, y
        target: (B, T, 2) ground truth x, y
    Returns:
        fde: scalar
    """
    final_displacement = torch.norm(pred[:, -1, :2] - target[:, -1, :2], dim=-1)  # (B,)
    fde = final_displacement.mean()
    return fde


def compute_mr(pred: torch.Tensor, target: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    """
    Miss Rate: fraction of predictions with FDE > threshold.
    Args:
        pred: (B, T, 2)
        target: (B, T, 2)
        threshold: distance threshold in meters
    Returns:
        mr: scalar
    """
    fde_per_sample = torch.norm(pred[:, -1, :2] - target[:, -1, :2], dim=-1)  # (B,)
    miss = (fde_per_sample > threshold).float()
    return miss.mean()


def compute_heading_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean heading (yaw) error.
    Args:
        pred: (B, T, 3) - last dim is yaw
        target: (B, T, 3)
    Returns:
        heading_error: scalar
    """
    yaw_error = torch.abs(pred[:, :, 2] - target[:, :, 2])
    # Normalize to [-pi, pi]
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
    return yaw_error.abs().mean()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all metrics for a batch."""
    return {
        'ADE': compute_ade(pred, target).item(),
        'FDE': compute_fde(pred, target).item(),
        'MR': compute_mr(pred, target).item(),
        'HeadingError': compute_heading_error(pred, target).item()
    }
