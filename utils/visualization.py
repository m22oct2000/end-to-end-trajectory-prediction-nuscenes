"""
Visualization utilities for trajectory prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional


def visualize_trajectory(
    past_traj: np.ndarray,
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Trajectory Prediction"
):
    """
    Visualize past, predicted, and ground truth trajectories.
    Args:
        past_traj: (T_past, 3) - x, y, yaw
        pred_traj: (T_future, 3)
        gt_traj: (T_future, 3)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(past_traj[:, 0], past_traj[:, 1], 'b-o', label='Past', linewidth=2, markersize=5)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', label='Ground Truth', linewidth=2, markersize=5)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--o', label='Prediction', linewidth=2, markersize=5)

    ax.scatter([0], [0], c='black', s=100, zorder=5, label='Ego')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_metrics_over_time(metrics_history: dict, save_path: Optional[str] = None):
    """Plot training metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = list(metrics_history.keys())
    for i, (ax, key) in enumerate(zip(axes.flatten(), keys)):
        ax.plot(metrics_history[key])
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
