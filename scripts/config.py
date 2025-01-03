import os
from _utils import get_file_dir

## training
MIN_EPOCHS = 1
MAX_EPOCHS = 20
# ACCELERATOR = "gpu"
# DEVICES = 1
MULTI_GPU_STRATEGY="auto"
LEARNING_RATE =0.02
TRAIN_PRECISION="16-mixed"
NMS_THRESH = 0.5 ## threshold for non-max suppression
ACCUM_GRAD_BATCHES = 4
## detector
ANCHOR_SIZES = ((32,64,128,256,512,),) * 1
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)

TRAINABLE_BACKBONE_LAYERS = 3
NUM_CLASSES = 23 + 1 ## +1 for background
BACKBONE_LOAD_PATH = os.path.join(get_file_dir(), "../trained_backbones/mobilenet_v3_large=val_f1score0.99.pt")

## data
# ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Train_no_dup_bboxes.csv")
ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Train.csv")
PRED_ANNOT_FILEPATH = os.path.join(get_file_dir(), "../data/Test.csv")
IMGS_PATH = os.path.join(get_file_dir(), "../data/images")
NUM_WORKERS = 0
BATCH_SIZE = 8
PERSISTENT_WORKERS=False

## others
LOGS_PATH = os.path.join(get_file_dir(), "../tb_logs")
CHECKPOINT_SAVEPATH = os.path.join(get_file_dir(), "../checkpoints")

CHECKPOINT_LOAD_PATH = os.path.join(get_file_dir(), "../model/fcnn_crop_disease_statedict.ckpt")
# CHECKPOINT_LOAD_PATH = None
SEED=42

## metrics
METRICS_SAVE_PATH = os.path.join(get_file_dir(), "../metrics/map_per_class.csv")

## submissions
SUBMISSION_PATH = os.path.join(get_file_dir(), "../submissions")