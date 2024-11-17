import lightning as L
import pandas as pd
from torch.utils.data import random_split, DataLoader
from datasets import ImageDataset, ImagePredictionDataset
import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


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
        persistent_workers=False,
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


    def _split_data(self):
        df = pd.read_csv(self.annotations_filepath)
        df = df.drop_duplicates(subset="Image_ID", keep="last", ignore_index=True)
        train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df["class"], random_state=self.seed)
        val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["class"], random_state=self.seed)

        return train_df, val_df, test_df

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage != "predict":
            train_df, val_df, test_df = self._split_data()

            self.train_dataset = ImageDataset(train_df, self.imgs_path, self.label_encoder, transforms=self.transforms)
            self.val_dataset = ImageDataset(val_df, self.imgs_path, self.label_encoder, transforms=self.transforms)
            self.test_dataset = ImageDataset(test_df, self.imgs_path, self.label_encoder, transforms=self.transforms)

        ## for preds
        if stage == "predict":
            self.pred_dataset = ImagePredictionDataset(
                self.pred_annotations_filepath,
                self.imgs_path,
                transforms=self.transforms,
            )

    def _collate_wrapper(self, batch):
        images, targets = list(zip(*batch))
        images = list(images)
        targets = list(targets)
        if self.pin_memory:
            for i in range(len(images)):
                images[i] = images[i].pin_memory()
                targets[i]["boxes"] = targets[i]["boxes"].pin_memory()
                targets[i]["labels"] = targets[i]["labels"].pin_memory()

        return images, targets

    def _pred_collate_wrapper(self, batch):
        return list(zip(*batch))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_wrapper,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._pred_collate_wrapper,
            persistent_workers=self.persistent_workers,
        )
