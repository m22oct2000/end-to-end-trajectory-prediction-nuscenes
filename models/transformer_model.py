"""
Main transformer model for End-to-End trajectory prediction.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .encoders.image_encoder import ImageEncoder
from .encoders.lidar_encoder import PointNetEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_features, key_features, value_features=None):
        if value_features is None:
            value_features = key_features
        attended, attn_weights = self.cross_attn(query_features, key_features, value_features)
        output = self.norm(query_features + attended)
        return output, attn_weights


class MultiModalTransformer(nn.Module):
    """End-to-End transformer for trajectory prediction."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config['model']['d_model']
        nhead = config['model']['nhead']
        num_layers = config['model']['num_encoder_layers']
        dim_feedforward = config['model']['dim_feedforward']
        dropout = config['model']['dropout']

        self.image_encoder = ImageEncoder(
            backbone=config['encoders']['image']['backbone'],
            pretrained=config['encoders']['image']['pretrained'],
            output_dim=d_model
        )
        self.lidar_encoder = PointNetEncoder(
            input_dim=3,
            hidden_dim=config['encoders']['lidar']['pointnet_features'],
            output_dim=d_model
        )

        traj_input_dim = config['encoders']['trajectory']['input_dim']
        traj_hidden = config['encoders']['trajectory']['hidden_dim']
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(traj_input_dim, traj_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(traj_hidden, d_model)
        )

        self.image_lidar_fusion = CrossModalFusion(d_model, nhead, dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=config['model']['max_seq_len'], dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        future_frames = config['data']['prediction_horizon']
        output_dim = future_frames * 3
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, camera_images, lidar_points, past_trajectory):
        batch_size, seq_len = camera_images.shape[:2]
        image_features = self.image_encoder(camera_images)
        lidar_features = self.lidar_encoder(lidar_points)
        traj_features = self.trajectory_encoder(past_trajectory)
        fused_features, attn_weights = self.image_lidar_fusion(image_features, lidar_features)
        fused_features = fused_features + traj_features
        fused_features = self.pos_encoder(fused_features)
        temporal_features = self.temporal_transformer(fused_features)
        last_features = temporal_features[:, -1, :]
        future_flat = self.trajectory_head(last_features)
        future_frames = self.config['data']['prediction_horizon']
        future_trajectory = future_flat.view(batch_size, future_frames, 3)
        return future_trajectory, attn_weights
