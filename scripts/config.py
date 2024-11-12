import os
from utils import get_file_dir

## training
MIN_EPOCHS = 1
MAX_EPOCHS = 10
ACCELERATOR = "cpu"
DEVICES = 1

## data
ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Train.csv")
PRED_ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Test.csv")
IMGS_PATH = os.path.join(get_file_dir(), "../data/Train.csv")
NUM_WORKERS = 4
BATCH_SIZE = 4

## others
LOGS_PATH = "../tb_logs"
CHECKPOINT_SAVEPATH = "../checkpoints/"
BACKBONE_PATH = os.path.join(
    get_file_dir(), "../trained_backbones/mobilenet_v3_large=0.80.pt"
)
