import lightning as L
from model import GCDDDetector
from data_module import DetectionDataModule
from callbacks import get_callbacks
import config
import torch
import os
from utils import get_labelencoder, get_transforms, get_test_transforms, get_num_classes
from lightning.pytorch.loggers import TensorBoardLogger
import sys
from argparse import ArgumentParser


if __name__ == "__main__":
    ## getting scripts arguments
    parser = ArgumentParser()
    ## training args
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--precision", type=str, default="16-mixed")

    ## data args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", type=bool, default=False)
    parser.add_argument("--pin_memory", type=int, default=1)

    ## run args
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--test_on_train_data", action="store_true", default=False)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=config.CHECKPOINT_LOAD_PATH)

    args = parser.parse_args()
    
    ## getting paths
    annot_filepath = os.environ.get("GCDD_ANNOT_FILEPATH") or config.ANNOT_FILEPATH
    pred_annot_filepath = (
        os.environ.get("GCDD_PRED_ANNOT_FILEPATH") or config.PRED_ANNOT_FILEPATH
    )

    logs_path = os.environ.get("GCDD_LOGS_PATH") or config.LOGS_PATH
    imgs_path = os.environ.get("GCDD_IMGS_PATH") or config.IMGS_PATH

    num_classes = get_num_classes(annot_filepath, "class")
    label_encoder = get_labelencoder(annot_filepath, "class")
    
    ## logger
    logger = TensorBoardLogger(config.LOGS_PATH, "gcdd_v0")

    ## instantiation
    data_module = DetectionDataModule(
        annot_filepath,
        pred_annotations_filepath=pred_annot_filepath,
        imgs_path=imgs_path,
        label_encoder=label_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory == 1,
        seed=config.SEED,
        transforms=get_transforms(),
        test_transforms=get_test_transforms()
    )

    if args.load_checkpoint and args.checkpoint_path:
        model = GCDDDetector.load_from_checkpoint(config.CHECKPOINT_LOAD_PATH,
                                                  num_classes=num_classes,
                                                  learning_rate=args.learning_rate,
                                                  # trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS
                                                 )
        print("CHECKPOINT LOADED")
    else:
        model = GCDDDetector(num_classes,
                             learning_rate=args.learning_rate
                             # trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS
                            )
        print("NO CHECKPOINT BEING USED")
    # model = torch.compile(model, dynamic=True)

    
    trainer = L.Trainer(
        accumulate_grad_batches=config.ACCUM_GRAD_BATCHES,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=config.MULTI_GPU_STRATEGY,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=get_callbacks(),
        logger=logger,
        fast_dev_run=args.fast_dev_run,        
    )

    ##
    try:
        mode = args.mode
    except:
        raise Exception("Please provide a system argument--train/test/validate. eg. pythong train.py [train/validate/test]")

    if mode == "train":
        # print(f"TRAINING MODEL AT LEARNING RATE OF: {args.learning_rate}")
        trainer.fit(model, data_module)
    elif mode == "validate":
        trainer.validate(model, data_module)
    elif mode == "test":
        if args.test_on_train_data:
            data_module.setup("test_on_train")
            trainer.test(model, datamodule=data_module)
        else:
            trainer.test(model, data_module)
    elif mode == "predict":
        trainer.predict(model, data_module)
