"""
nuScenes dataset constants and coordinate system definitions.
"""

# Sensor channels
CAMERA_CHANNELS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT'
]

LIDAR_CHANNEL = 'LIDAR_TOP'
RADAR_CHANNELS = ['RADAR_FRONT', 'RADAR_BACK']

# Coordinate frames
WORLD_FRAME = 'world'
EGO_FRAME = 'ego'
SENSOR_FRAME = 'sensor'

# Trajectory parameters
HISTORY_SECONDS = 5
FUTURE_SECONDS = 3
FPS = 2
HISTORY_FRAMES = HISTORY_SECONDS * FPS  # 10 frames
FUTURE_FRAMES = FUTURE_SECONDS * FPS     # 6 frames

# Trajectory output dimensions
TRAJ_DIM = 3
TRAJ_FEATURES = 10

# Image preprocessing
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# LiDAR preprocessing
LIDAR_POINTS = 1024
LIDAR_RANGE = [-50, 50]

# Dataset splits
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
