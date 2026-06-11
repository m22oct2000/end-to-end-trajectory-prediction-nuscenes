"""
PyTorch Dataset class for trajectory prediction.
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for End-to-End trajectory prediction."""

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        data_path = self.data_dir / f'processed_{split}.pkl'
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} samples from {split} split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        camera = torch.FloatTensor(sample['camera_image'])
        lidar = torch.FloatTensor(sample['lidar_points'])
        past_traj = torch.FloatTensor(sample['past_trajectory'])
        future_traj = torch.FloatTensor(sample['future_trajectory'])

        lidar = lidar / 50.0
        past_traj[:, :2] = past_traj[:, :2] - past_traj[0, :2]
        future_traj[:, :2] = future_traj[:, :2] - past_traj[0, :2]

        if self.transform:
            camera = self.transform(camera)

        return {
            'camera': camera,
            'lidar': lidar,
            'past_trajectory': past_traj,
            'future_trajectory': future_traj,
            'sample_token': sample['sample_token']
        }


def collate_fn(batch):
    cameras = torch.stack([item['camera'] for item in batch])
    lidars = torch.stack([item['lidar'] for item in batch])
    past_trajs = torch.stack([item['past_trajectory'] for item in batch])
    future_trajs = torch.stack([item['future_trajectory'] for item in batch])
    tokens = [item['sample_token'] for item in batch]
    return {
        'camera': cameras,
        'lidar': lidars,
        'past_trajectory': past_trajs,
        'future_trajectory': future_trajs,
        'sample_token': tokens
    }
