import os
from utils import get_file_dir

## training
MIN_EPOCHS = 1
MAX_EPOCHS = 20
ACCELERATOR = "gpu"
DEVICES = 1
MULTI_GPU_STRATEGY="auto"
LEARNING_RATE =0.0002
BATCHED_NMS_THRESH = 0.5

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
CHECKPOINT_LOAD_PATH = os.path.join(get_file_dir(), "../checkpoints/epoch-epoch=00_lr=0.0004_map@50=map_50=0.22.ckpt")
