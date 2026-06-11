"""
Unit tests for evaluation metrics.
"""

import torch
import pytest
from training.metrics import compute_ade, compute_fde, compute_mr


def test_ade_perfect():
    pred = torch.zeros(4, 6, 3)
    target = torch.zeros(4, 6, 3)
    assert compute_ade(pred, target).item() == pytest.approx(0.0)


def test_fde_perfect():
    pred = torch.zeros(4, 6, 3)
    target = torch.zeros(4, 6, 3)
    assert compute_fde(pred, target).item() == pytest.approx(0.0)


def test_mr_all_miss():
    pred = torch.zeros(4, 6, 3)
    target = torch.ones(4, 6, 3) * 10.0
    mr = compute_mr(pred, target, threshold=2.0)
    assert mr.item() == pytest.approx(1.0)


def test_mr_no_miss():
    pred = torch.zeros(4, 6, 3)
    target = torch.zeros(4, 6, 3)
    mr = compute_mr(pred, target, threshold=2.0)
    assert mr.item() == pytest.approx(0.0)
