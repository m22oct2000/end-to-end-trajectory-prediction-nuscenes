from setuptools import setup, find_packages

setup(
    name='trajectory-prediction',
    version='0.1.0',
    description='End-to-End Trajectory Prediction for Autonomous Driving using nuScenes',
    author='m22oct2000',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'nuscenes-devkit>=1.1.10',
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'einops>=0.6.1',
        'timm>=0.9.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'wandb>=0.15.0',
    ]
)
