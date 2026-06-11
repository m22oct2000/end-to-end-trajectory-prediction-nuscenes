"""
LiDAR encoder using PointNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """PointNet encoder for LiDAR point clouds."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        single_input = False
        if x.dim() == 3:
            single_input = True
            x = x.unsqueeze(1)

        batch_size, seq_len, N, input_dim = x.shape
        x = x.view(batch_size * seq_len, N, input_dim)
        x = self.mlp1(x)
        x = torch.max(x, dim=1)[0]
        x = self.global_mlp(x)
        x = x.view(batch_size, seq_len, -1)
        if single_input:
            x = x.squeeze(1)
        return x
