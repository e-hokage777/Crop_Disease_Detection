import torch
from skimage import io
import pandas as pd
import os
import numpy as np
from torchvision import tv_tensors
from torchvision.io import decode_image


## DataLoader for loading images for detection
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        label_encoder,
        transforms=None,
    ):

        self.df = annotations_file
        self.transforms = transforms
        self.class_encoder = label_encoder

        ## encoding the class labels
        self.df["class"] = self.class_encoder.transform(self.df["class"])

        ## getting image names
        self.img_labels = self.df["Image_ID"].unique()
        self.targets = self.process_target()
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = decode_image(img_path)

        target = self.targets[self.img_labels[idx]].copy()
        
        if self.transforms:
            # image = tv_tensors.Image(image, dtype=torch.float32)
            target["boxes"] = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=image.shape[-2:])
            image, target = self.transforms(image, target)
            # print(image.max(), image.min())

        return image, target

    def process_target(self):
        image_map = dict()
        for group_name, group_df in self.df.groupby("Image_ID"):
            image_map[group_name] = {
                "boxes": torch.from_numpy(
                    group_df[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
                ),
                "labels": torch.from_numpy(group_df["class"].values.astype(np.int64)),
            }

            ## taking care of situation where line instead of boxes were drawn
            for i in range(len(image_map[group_name]["boxes"])):
                if (
                    image_map[group_name]["boxes"][i][0]
                    == image_map[group_name]["boxes"][i][2]
                ):
                    image_map[group_name]["boxes"][i][2] += 1
                if (
                    image_map[group_name]["boxes"][i][1]
                    == image_map[group_name]["boxes"][i][3]
                ):
                    image_map[group_name]["boxes"][i][3] += 1

        return image_map


class ImagePredictionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_filepath, imgs_path, transforms=None):
        super().__init__()
        self.imgs_path = imgs_path
        self.transforms = transforms

        df = pd.read_csv(annotations_filepath)
        self.img_labels = df["Image_ID"].unique()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.imgs_path, self.img_labels[idx]))
        image_name = self.img_labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image_name, image
