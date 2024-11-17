import os
from utils import get_file_dir

## training
MIN_EPOCHS = 1
MAX_EPOCHS = 20
ACCELERATOR = "gpu"
DEVICES = 1
MULTI_GPU_STRATEGY="auto"
LEARNING_RATE =0.008
TRAIN_PRECISION="32-true"
NMS_THRESH = 0.01 ## threshold for non-max suppression
ACCUM_GRAD_BATCHES = 6

## detector
ANCHOR_SIZES = ((64, 128, 400, 512, 800),) * 3
ANCHOR_RATIOS = ((0.7, 1.0, 1.5),) * 3
TRAINABLE_BACKBONE_LAYERS = 3
NUM_CLASSES = 23 + 1 ## +1 for background

## data
ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Train.csv")
PRED_ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Test.csv")
IMGS_PATH = os.path.join(get_file_dir(), "../data/images")
NUM_WORKERS = 0
BATCH_SIZE = 10
PERSISTENT_WORKERS=False

## others
LOGS_PATH = os.path.join(get_file_dir(), "../tb_logs")
CHECKPOINT_SAVEPATH = os.path.join(get_file_dir(), "../checkpoints")
BACKBONE_PATH = os.path.join(
    get_file_dir(), "../trained_backbones/mobilenet_v3_large=0.80.pt"
)
# CHECKPOINT_LOAD_PATH = os.path.join(get_file_dir(), "../checkpoints/epoch-epoch=00_lr=0.0001_map@50=map_50=0.00-v1.ckpt")
# CHECKPOINT_LOAD_PATH = os.path.join(get_file_dir(), "../checkpoints/epoch-epoch=03_lr=0.0002_map@50=map_50=0.22.ckpt")
CHECKPOINT_LOAD_PATH = None
SEED=42

## submissions
SUBMISSION_PATH = os.path.join(get_file_dir(), "../submissions")