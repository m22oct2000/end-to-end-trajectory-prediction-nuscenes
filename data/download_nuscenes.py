"""
Download and setup nuScenes dataset.
Run this script first before anything else.
"""

import os
import subprocess
import argparse
from pathlib import Path

def download_nuscenes(root_dir: str, version: str = "v1.0-mini"):
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading nuScenes {version} to {root_dir}")

    if version == "v1.0-mini":
        print("""
        To download nuScenes:
        1. Go to https://www.nuscenes.org/download
        2. Create a free account
        3. Download the 'nuScenes mini' dataset (approx 5GB)
        4. Extract to: {}/nuscenes
        """.format(root_dir))
    else:
        print("Full dataset requires registration. Download from nuscenes.org")

    try:
        from nuscenes import NuScenes
        print("Verifying installation...")
        nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
        print(f"Successfully loaded {version} with {len(nusc.sample)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please manually download from nuscenes.org")

def setup_directory_structure(root_dir: str):
    dirs = [
        "samples/LIDAR_TOP",
        "samples/CAM_FRONT",
        "samples/CAM_FRONT_RIGHT",
        "samples/CAM_FRONT_LEFT",
        "samples/CAM_BACK",
        "maps",
        "v1.0-mini"
    ]
    for d in dirs:
        (Path(root_dir) / d).mkdir(parents=True, exist_ok=True)
    print("Directory structure created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    args = parser.parse_args()

    setup_directory_structure(args.root_dir)
    download_nuscenes(args.root_dir, args.version)

    print("""
    Next steps:
    1. Manually download nuScenes mini from nuscenes.org
    2. Extract to: {}
    3. Run: python data/preprocessing.py
    """.format(args.root_dir))
