"""
Unit tests for model components.
"""

import torch
import pytest


def get_test_config():
    return {
        'model': {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2,
                  'num_decoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.0, 'max_seq_len': 20},
        'encoders': {
            'image': {'backbone': 'resnet18', 'pretrained': False, 'output_dim': 64},
            'lidar': {'pointnet_features': 64, 'num_points': 64},
            'trajectory': {'input_dim': 10, 'hidden_dim': 32}
        },
        'data': {'prediction_horizon': 6}
    }


def test_image_encoder():
    from models.encoders.image_encoder import ImageEncoder
    enc = ImageEncoder(backbone='resnet18', pretrained=False, output_dim=64)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"


def test_lidar_encoder():
    from models.encoders.lidar_encoder import PointNetEncoder
    enc = PointNetEncoder(input_dim=3, hidden_dim=64, output_dim=64)
    x = torch.randn(2, 128, 3)
    out = enc(x)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"


def test_full_model():
    from models.transformer_model import MultiModalTransformer
    config = get_test_config()
    model = MultiModalTransformer(config)
    cameras = torch.randn(2, 1, 3, 224, 224)
    lidars = torch.randn(2, 1, 128, 3)
    past_traj = torch.randn(2, 10, 10)
    pred, attn = model(cameras, lidars, past_traj)
    assert pred.shape == (2, 6, 3), f"Expected (2, 6, 3), got {pred.shape}"
