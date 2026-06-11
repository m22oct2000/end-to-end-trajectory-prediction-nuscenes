"""
Evaluation script for trajectory prediction model.
"""

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import TrajectoryDataset, collate_fn
from models.transformer_model import MultiModalTransformer
from training.metrics import evaluate_batch
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_trajectory


def evaluate(config_path: str = 'config/config.yaml', checkpoint_path: str = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')

    test_ds = TrajectoryDataset(config['data']['root_dir'], split='test')
    test_loader = DataLoader(test_ds, batch_size=config['data']['batch_size'],
                             shuffle=False, num_workers=config['data']['num_workers'],
                             collate_fn=collate_fn)

    model = MultiModalTransformer(config).to(device)
    if checkpoint_path:
        load_checkpoint(model, None, checkpoint_path)
    model.eval()

    all_metrics = {'ADE': [], 'FDE': [], 'MR': [], 'HeadingError': []}
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            cameras = batch['camera'].to(device).unsqueeze(1)
            lidars = batch['lidar'].to(device).unsqueeze(1)
            past_traj = batch['past_trajectory'].to(device)
            future_traj = batch['future_trajectory'].to(device)

            pred, _ = model(cameras, lidars, past_traj)
            metrics = evaluate_batch(pred, future_traj)

            for k, v in metrics.items():
                all_metrics[k].append(v)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(future_traj.cpu().numpy())

    print("\n=== Evaluation Results ===")
    for k, v in all_metrics.items():
        print(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    return all_metrics, np.concatenate(all_preds), np.concatenate(all_targets)


if __name__ == '__main__':
    evaluate()
