## training
MIN_EPOCHS = 1
MAX_EPOCHS = 10
ACCELERATOR = "gpu"
DEVICES = 1

## data
ANNOT_FILEPATH = "../data/Train.csv"
PRED_ANNOT_FILEPATH = "../data/Test.csv"
IMGS_PATH = "../images"
NUM_WORKERS = 4
BATCH_SIZE = 4

## others
LOGS_PATH = "../tb_logs"
CHECKPOINT_SAVEPATH = "../checkpoints/"
BACKBONE_PATH = "../trained_backbones/mobilenet_v3_large=0.80.pt"
