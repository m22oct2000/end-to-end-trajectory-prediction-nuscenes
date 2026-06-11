# End-to-End Trajectory Prediction for Autonomous Driving

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Multimodal transformer architecture that fuses camera images, LiDAR point clouds, and past ego-vehicle trajectory to predict 3 seconds into the future on the nuScenes dataset.

---

## Architecture

```
Camera (ResNet50 / ViT) ──┐
                          ├─► Cross-Modal Attention ──► Temporal Transformer ──► Trajectory Head ──► (6×3) Future
LiDAR (PointNet) ─────────┘
Past Trajectory (MLP) ────────────────────────────────────────────────────────────────────────────┘
```

| Component | Detail |
|-----------|--------|
| Image Encoder | ResNet-50 (pretrained) → Linear(2048→256) |
| LiDAR Encoder | PointNet → Linear(512→256) |
| Fusion | Cross-attention (8 heads, d=256) |
| Temporal | 4-layer Transformer Encoder |
| Output | 6 timesteps × (x, y, yaw) |

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/m22oct2000/end-to-end-trajectory-prediction-nuscenes
cd end-to-end-trajectory-prediction-nuscenes
pip install -e .

# 2. Download nuScenes mini (register at nuscenes.org)
bash scripts/download_data.sh

# 3. Preprocess
python data/preprocessing.py

# 4. Train
bash scripts/run_training.sh

# 5. Evaluate
bash scripts/run_evaluation.sh outputs/checkpoints/checkpoint_best.pt
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **ADE** | Average Displacement Error (m) over all 6 timesteps |
| **FDE** | Final Displacement Error (m) at t=3s |
| **MR** | Miss Rate — % samples with FDE > 2m |
| **Heading Error** | Mean absolute yaw error (rad) |

---

## Dataset

Uses [nuScenes](https://www.nuscenes.org/) (Boston + Singapore). Start with `v1.0-mini` (~5 GB) for fast iteration, then scale to `v1.0-trainval` for full training.

---

## Project Structure

```
├── config/          # Hyperparameters (YAML)
├── data/            # Dataset, preprocessing, constants
├── models/          # Transformer + encoders
├── training/        # Train loop, eval, metrics, losses
├── utils/           # Logger, checkpoints, visualization
├── notebooks/       # EDA, visualization, failure analysis
├── scripts/         # One-click shell scripts
└── tests/           # Unit tests
```

---

## License

MIT
