import lightning as L
import pandas as pd
from torch.utils.data import random_split, DataLoader
from datasets import ImageDataset, ImagePredictionDataset
import torch
from torchvision.transforms import ToTensor


class DetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        annotations_filepath,
        pred_annotations_filepath,
        label_encoder,
        imgs_path,
        num_workers=4,
        batch_size=4,
        split_ratio=[0.7, 0.15, 0.15],
        seed=42,
        pin_memory=True,
        persistent_workers=True,
        transforms=ToTensor(),
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.annotations_filepath = annotations_filepath
        self.pred_annotations_filepath = pred_annotations_filepath
        self.imgs_path = imgs_path
        self.label_encoder = label_encoder
        self.split_ratio = split_ratio
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.transforms = transforms

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage != "predict":
            dataset = ImageDataset(
                self.annotations_filepath,
                self.imgs_path,
                self.label_encoder,
                transforms=self.transforms,
                target_transforms=None,
            )

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, self.split_ratio, generator=generator
            )

        ## for preds
        if stage == "predict":
            self.pred_dataset = ImagePredictionDataset(
                self.pred_annotations_filepath,
                self.imgs_path,
                transforms=self.transofrms,
            )

    def _collate_wrapper(self, batch):
        images, targets = list(zip(*batch))

        if self.pin_memory:
            for i in range(len(images)):
                images[i].pin_memory()
                targets[i]["boxes"].pin_memory()
                targets[i]["labels"].pin_memory()

        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_wrapper,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,
            persisten_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,
            persisten_workers=self.persistent_workers,
        )
