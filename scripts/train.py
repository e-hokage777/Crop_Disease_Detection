import lightning as L
from model import GCDDDetector
from data_module import DetectionDataModule
from callbacks import get_callbacks
import config
import torch
import os
from utils import get_labelencoder


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

    ## instantiation
    data_module = DetectionDataModule(
        annot_filepath,
        pred_annotations_filepath=pred_annot_filepath,
        imgs_path=imgs_path,
        label_encoder=label_encoder,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        persistent_workers=True,
    )

    model = GCDDDetector(num_classes, learning_rate=0.001)
    # model = torch.compile(model, dynamic=True)

    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        devices=config.DEVICES,
        callbacks=get_callbacks(),
        fast_dev_run=True,
    )

    trainer.fit(model, data_module)
