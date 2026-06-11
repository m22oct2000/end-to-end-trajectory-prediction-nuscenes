"""
Checkpoint save/load utilities.
"""

import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, tag='latest'):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics
    }
    path = Path(checkpoint_dir) / f'checkpoint_{tag}.pt'
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    if optimizer and state.get('optimizer_state_dict'):
        optimizer.load_state_dict(state['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {state.get('epoch')}")
    return state.get('epoch'), state.get('metrics')
