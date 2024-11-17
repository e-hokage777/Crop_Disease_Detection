import lightning as L
from model import GCDDDetector
from data_module import DetectionDataModule
from callbacks import get_callbacks
import config
import torch
import os
from utils import get_labelencoder
from lightning.pytorch.loggers import TensorBoardLogger
import sys


if __name__ == "__main__":
    ## getting paths
    annot_filepath = os.environ.get("GCDD_ANNOT_FILEPATH") or config.ANNOT_FILEPATH
    pred_annot_filepath = (
        os.environ.get("GCDD_PRED_ANNOT_FILEPATH") or config.PRED_ANNOT_FILEPATH
    )

    logs_path = os.environ.get("GCDD_LOGS_PATH") or config.LOGS_PATH
    imgs_path = os.environ.get("GCDD_IMGS_PATH") or config.IMGS_PATH

    num_classes = 23 + 1
    label_encoder = get_labelencoder(annot_filepath, "class")
    
    ## logger
    logger = TensorBoardLogger(config.LOGS_PATH, "gcdd_v0")

    ## instantiation
    data_module = DetectionDataModule(
        annot_filepath,
        pred_annotations_filepath=pred_annot_filepath,
        imgs_path=imgs_path,
        label_encoder=label_encoder,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        seed=config.SEED
    )

    if config.CHECKPOINT_LOAD_PATH:
        model = GCDDDetector.load_from_checkpoint(config.CHECKPOINT_LOAD_PATH,
                                                  num_classes=num_classes,
                                                  learning_rate=config.LEARNING_RATE,
                                                  trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS
                                                 )
        print("CHECKPOINT LOADED")
    else:
        model = GCDDDetector(num_classes,
                             learning_rate=config.LEARNING_RATE,
                             trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS
                            )
        print("NO CHECKPOINT BEING USED")
    # model = torch.compile(model, dynamic=True)

    
    trainer = L.Trainer(
        accumulate_grad_batches=config.ACCUM_GRAD_BATCHES,
        accelerator=config.ACCELERATOR,
        strategy=config.MULTI_GPU_STRATEGY,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        devices=config.DEVICES,
        precision=config.TRAIN_PRECISION,
        callbacks=get_callbacks(),
        logger=logger,
    )

    ##
    try:
        mode = sys.argv[1]
    except:
        raise Exception("Please provide a system argument--train/test/validate. eg. pythong train.py [train/validate/test]")

    if mode == "train":
        print(f"TRAINING MODEL AT LEARNING RATE OF: {config.LEARNING_RATE}")
        trainer.fit(model, data_module)
    elif mode == "validate":
        trainer.validate(model, data_module)
    elif mode == "test":
        trainer.test(model, data_module)
    elif mode == "predict":
        trainer.predict(model, data_module)
