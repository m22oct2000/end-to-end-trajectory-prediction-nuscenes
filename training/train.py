"""
Training loop for trajectory prediction model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import wandb

from data.dataset import TrajectoryDataset, collate_fn
from models.transformer_model import MultiModalTransformer
from training.losses import TrajectoryLoss
from training.metrics import evaluate_batch
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, config, epoch):
    model.train()
    total_loss = 0.0
    log_interval = config['system']['log_interval']

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        cameras = batch['camera'].to(device)        # (B, 3, H, W) — single frame
        lidars = batch['lidar'].to(device)           # (B, N, 3)
        past_traj = batch['past_trajectory'].to(device)  # (B, T_past, D)
        future_traj = batch['future_trajectory'].to(device)  # (B, T_future, 3)

        # Add seq dimension for single-frame input
        cameras = cameras.unsqueeze(1)   # (B, 1, C, H, W)
        lidars = lidars.unsqueeze(1)     # (B, 1, N, 3)

        with autocast(enabled=config['system']['mixed_precision']):
            pred, _ = model(cameras, lidars, past_traj)
            loss, xy_loss, heading_loss = criterion(pred, future_traj)
            loss = loss / config['training']['accumulation_steps']

        scaler.scale(loss).backward()

        if (step + 1) % config['training']['accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config['training']['accumulation_steps']

        if step % log_interval == 0:
            wandb.log({'train/loss': loss.item(), 'train/xy_loss': xy_loss.item(),
                       'train/heading_loss': heading_loss.item(), 'epoch': epoch, 'step': step})

    return total_loss / len(loader)


def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_metrics = {'ADE': 0.0, 'FDE': 0.0, 'MR': 0.0, 'HeadingError': 0.0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            cameras = batch['camera'].to(device).unsqueeze(1)
            lidars = batch['lidar'].to(device).unsqueeze(1)
            past_traj = batch['past_trajectory'].to(device)
            future_traj = batch['future_trajectory'].to(device)

            pred, _ = model(cameras, lidars, past_traj)
            loss, _, _ = criterion(pred, future_traj)
            total_loss += loss.item()

            metrics = evaluate_batch(pred, future_traj)
            for k, v in metrics.items():
                all_metrics[k] += v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['system']['seed'])

    wandb.init(project="trajectory-prediction", config=config)
    logger = setup_logger(config['paths']['logs'])

    train_ds = TrajectoryDataset(config['data']['root_dir'], split='train')
    val_ds = TrajectoryDataset(config['data']['root_dir'], split='val')
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'],
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'],
                            collate_fn=collate_fn)

    model = MultiModalTransformer(config).to(device)
    criterion = TrajectoryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                            weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    scaler = GradScaler()

    best_ade = float('inf')
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config, epoch)
        if epoch % config['system']['eval_interval'] == 0:
            val_loss, metrics = validate(model, val_loader, criterion, device, config)
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"ADE: {metrics['ADE']:.4f} | FDE: {metrics['FDE']:.4f}")
            wandb.log({'val/loss': val_loss, **{f'val/{k}': v for k, v in metrics.items()}, 'epoch': epoch})

            if metrics['ADE'] < best_ade:
                best_ade = metrics['ADE']
                save_checkpoint(model, optimizer, epoch, metrics, config['paths']['checkpoints'], tag='best')

        scheduler.step()
        if epoch % config['system']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, {}, config['paths']['checkpoints'], tag=f'epoch_{epoch}')

    wandb.finish()
    print(f"Training complete. Best ADE: {best_ade:.4f}")


if __name__ == '__main__':
    main()
