#!/bin/bash
set -e
echo "Setting up nuScenes data directory..."
python data/download_nuscenes.py --root_dir ./data/nuscenes --version v1.0-mini
echo "Done. Follow instructions above to download the dataset manually."
