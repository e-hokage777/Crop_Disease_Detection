import lightning as L
import pandas as pd
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from datasets import ImageDataset, ImagePredictionDataset
import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from _utils import clean_bbox_data
from collections import Counter

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
        test_transforms=ToTensor()
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
        self.test_transforms = test_transforms



    def _clean_train_df(self, df):
        return clean_bbox_data(df, remove_dups=False, remove_outs=False)
        
    def _split_data(self, df):
        df_unique = df.drop_duplicates(subset="Image_ID", keep="first", ignore_index=True)
        train_df, test_val_df = train_test_split(df_unique, test_size=0.2, stratify=df_unique["class"], random_state=self.seed)

        train_df = df[df["Image_ID"].isin(train_df["Image_ID"])].reset_index(drop=True)
        test_val_df = df[df["Image_ID"].isin(test_val_df["Image_ID"])].reset_index(drop=True)

        self.df_unique = df_unique


        return train_df, test_val_df

    def _get_weighted_sampler(self, dataset):
        class_counts = Counter(self.label_encoder.transform(self.df_unique["class"]))

        
        sample_weights = [0] * len(dataset)
        
        for idx, img_label in enumerate(dataset.img_labels):
            class_idx = self.label_encoder.transform([self.df_unique[self.df_unique["Image_ID"]==img_label]["class"].values[0]])[0]
            sample_weights[idx]  = 1/class_counts[class_idx]

        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        


    def setup(self, stage):
        if stage != "predict":
            df = pd.read_csv(self.annotations_filepath)
            train_df, val_test_df = self._split_data(df)

            ## cleaning train data (CLEANING SHOULD ONLY BE DONE ON TRAINING DATA)
            train_df = self._clean_train_df(train_df)

            self.train_dataset = ImageDataset(train_df, self.imgs_path, self.label_encoder, transforms=self.transforms)
            self.val_test_dataset = ImageDataset(val_test_df, self.imgs_path, self.label_encoder, transforms=self.test_transforms)
            self.train_sampler = self._get_weighted_sampler(self.train_dataset)


        ## for preds
        if stage == "predict":
            self.pred_dataset = ImagePredictionDataset(
                self.pred_annotations_filepath,
                self.imgs_path,
                transforms=self.test_transforms,
            )

    def _collate_wrapper(self, transforms=None):
        def func(batch):
            images, targets = list(zip(*batch))
            images = list(images)
            targets = list(targets)
    
            if torch.cuda.is_available() and self.pin_memory:
                for i in range(len(images)):
                    images[i] = images[i].pin_memory()
                    targets[i]["boxes"] = targets[i]["boxes"].pin_memory()
                    targets[i]["labels"] = targets[i]["labels"].pin_memory()
    
            if transforms:
                print("here")
                return transforms(images, targets)
    
            return images, targets

        return func

    def _pred_collate_wrapper(self, batch):
        return list(zip(*batch))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_wrapper(),
            persistent_workers=self.persistent_workers,
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper(),
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper(),
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
