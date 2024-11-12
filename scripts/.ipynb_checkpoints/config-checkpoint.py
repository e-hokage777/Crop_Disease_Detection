import os
from utils import get_file_dir

## training
MIN_EPOCHS = 1
MAX_EPOCHS = 10
ACCELERATOR = "gpu"
DEVICES = 1
MULTI_GPU_STRATEGY="auto"

## data
ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Train.csv")
PRED_ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Test.csv")
IMGS_PATH = os.path.join(get_file_dir(), "../data/images")
NUM_WORKERS = 0
BATCH_SIZE = 4
PERSISTENT_WORKERS=False

## others
LOGS_PATH = os.path.join(get_file_dir(), "../tb_logs")
CHECKPOINT_SAVEPATH = os.path.join(get_file_dir(), "../checkpoints")
BACKBONE_PATH = os.path.join(
    get_file_dir(), "../trained_backbones/mobilenet_v3_large=0.80.pt"
)
