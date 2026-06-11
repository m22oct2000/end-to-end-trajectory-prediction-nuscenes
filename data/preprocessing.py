"""
Preprocess nuScenes data for trajectory prediction.
Extracts camera images, LiDAR point clouds, and future trajectories.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from nuscenes import NuScenes
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import cv2
from tqdm import tqdm
import pickle

from .constants import *


class NuScenesPreprocessor:
    """Preprocess nuScenes dataset for trajectory prediction."""

    def __init__(self, root_dir: str, version: str = "v1.0-mini"):
        self.root_dir = Path(root_dir)
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(root_dir), verbose=False)
        self.samples = []

    def get_ego_trajectory(self, sample_token: str, future_frames: int = FUTURE_FRAMES):
        sample = self.nusc.get('sample', sample_token)
        current_sample = sample
        past_samples = []
        future_samples = []

        for _ in range(HISTORY_FRAMES):
            prev_token = current_sample['prev']
            if prev_token == '':
                break
            current_sample = self.nusc.get('sample', prev_token)
            past_samples.insert(0, current_sample)

        current_sample = sample

        for _ in range(future_frames):
            next_token = current_sample['next']
            if next_token == '':
                break
            current_sample = self.nusc.get('sample', next_token)
            future_samples.append(current_sample)

        def get_ego_pose(sample_data):
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            return np.array([ego_pose['translation'][0],
                             ego_pose['translation'][1],
                             ego_pose['translation'][2]]), Quaternion(ego_pose['rotation'])

        past_traj = []
        for s in past_samples:
            cam_front = self.nusc.get('sample_data', s['data']['CAM_FRONT'])
            pos, rot = get_ego_pose(cam_front)
            vel = pos - past_traj[-1][:3] if len(past_traj) > 0 else np.zeros(3)
            heading = rot.yaw_pitch_roll[0]
            past_traj.append(np.concatenate([pos, vel, [heading]]))

        while len(past_traj) < HISTORY_FRAMES:
            past_traj.insert(0, past_traj[0] if past_traj else np.zeros(10))

        future_traj = []
        for s in future_samples:
            cam_front = self.nusc.get('sample_data', s['data']['CAM_FRONT'])
            pos, rot = get_ego_pose(cam_front)
            heading = rot.yaw_pitch_roll[0]
            future_traj.append([pos[0], pos[1], heading])

        while len(future_traj) < future_frames:
            future_traj.append(future_traj[-1] if future_traj else [0, 0, 0])

        return np.array(past_traj[:HISTORY_FRAMES]), np.array(future_traj[:future_frames])

    def load_camera_image(self, sample_token: str, camera_name: str = 'CAM_FRONT') -> np.ndarray:
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][camera_name]
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = self.root_dir / cam_data['filename']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0
        img = (img - IMAGE_MEAN) / IMAGE_STD
        return img.transpose(2, 0, 1)

    def load_lidar_pc(self, sample_token: str) -> np.ndarray:
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        pc_path = self.root_dir / lidar_data['filename']
        pc = PointCloud.from_file(str(pc_path))
        points = pc.points[:3, :].T
        mask = (np.abs(points[:, 0]) < LIDAR_RANGE[1]) & \
               (np.abs(points[:, 1]) < LIDAR_RANGE[1]) & \
               (points[:, 2] > -2) & (points[:, 2] < 5)
        points = points[mask]
        if len(points) > LIDAR_POINTS:
            idx = np.random.choice(len(points), LIDAR_POINTS, replace=False)
            points = points[idx]
        else:
            padding = np.zeros((LIDAR_POINTS - len(points), 3))
            points = np.vstack([points, padding])
        return points

    def process_sample(self, sample_token: str) -> Dict:
        try:
            past_traj, future_traj = self.get_ego_trajectory(sample_token)
            camera_image = self.load_camera_image(sample_token)
            lidar_pc = self.load_lidar_pc(sample_token)
            return {
                'sample_token': sample_token,
                'camera_image': camera_image,
                'lidar_points': lidar_pc,
                'past_trajectory': past_traj,
                'future_trajectory': future_traj,
            }
        except Exception as e:
            print(f"Error processing {sample_token}: {e}")
            return None

    def create_dataset(self, split: str = 'train', max_samples: int = None):
        print(f"Processing {split} split...")
        if split == 'train':
            scene_tokens = self.nusc.scene[:int(0.7 * len(self.nusc.scene))]
        elif split == 'val':
            scene_tokens = self.nusc.scene[int(0.7 * len(self.nusc.scene)):int(0.85 * len(self.nusc.scene))]
        else:
            scene_tokens = self.nusc.scene[int(0.85 * len(self.nusc.scene)):]

        samples = []
        for scene_token in tqdm(scene_tokens):
            scene = self.nusc.get('scene', scene_token)
            current_token = scene['first_sample_token']
            while current_token != '':
                sample = self.process_sample(current_token)
                if sample:
                    samples.append(sample)
                sample_data = self.nusc.get('sample', current_token)
                current_token = sample_data['next']
                if max_samples and len(samples) >= max_samples:
                    break
            if max_samples and len(samples) >= max_samples:
                break

        save_path = self.root_dir / f'processed_{split}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved {len(samples)} samples to {save_path}")
        return samples


if __name__ == "__main__":
    preprocessor = NuScenesPreprocessor("./data/nuscenes", "v1.0-mini")
    train_data = preprocessor.create_dataset('train', max_samples=1000)
    val_data = preprocessor.create_dataset('val', max_samples=200)
    test_data = preprocessor.create_dataset('test', max_samples=100)
    print("Preprocessing complete!")
