"""
Unit tests for data pipeline.
"""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_collate_fn():
    from data.dataset import collate_fn
    batch = [
        {
            'camera': torch.randn(3, 224, 224),
            'lidar': torch.randn(1024, 3),
            'past_trajectory': torch.randn(10, 10),
            'future_trajectory': torch.randn(6, 3),
            'sample_token': 'tok1'
        },
        {
            'camera': torch.randn(3, 224, 224),
            'lidar': torch.randn(1024, 3),
            'past_trajectory': torch.randn(10, 10),
            'future_trajectory': torch.randn(6, 3),
            'sample_token': 'tok2'
        }
    ]
    out = collate_fn(batch)
    assert out['camera'].shape == (2, 3, 224, 224)
    assert out['lidar'].shape == (2, 1024, 3)
    assert out['past_trajectory'].shape == (2, 10, 10)
    assert out['future_trajectory'].shape == (2, 6, 3)
    assert len(out['sample_token']) == 2
