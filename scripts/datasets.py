import torch
from skimage import io
import pandas as pd
import os
import numpy as np
from torchvision import tv_tensors
from torchvision.io import decode_image, read_image


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
        self.img_labels = sorted(self.img_labels)
        self.targets = self.process_target()
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        # image = read_image(img_path)
        image = decode_image(img_path)

        target = self.targets[self.img_labels[idx]].copy()
        
        if self.transforms:
            # image = tv_tensors.Image(image, dtype=torch.float32)
            target["boxes"] = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=image.shape[-2:])
            image, target = self.transforms(image, target)
            # print(image.max(), image.min())

        ## filtering no-area boxes
        target["boxes"], target["labels"] = self.filter_bounding_boxes(target["boxes"], target["labels"])
        
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

        
        return image_map

    def filter_bounding_boxes(self, boxes, labels, area_threshold=1e-4):
        """
        Filters out bounding boxes with an area below a threshold.
        
        Args:
            boxes (Tensor): Tensor of shape (N, 4), where each row is (x_min, y_min, x_max, y_max).
            labels (Tensor): Corresponding labels for the boxes.
            area_threshold (float): Minimum area threshold to keep a box.
            
        Returns:
            filtered_boxes, filtered_labels: Tensors of valid boxes and their labels.
        """
        # Compute box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Filter boxes with area greater than the threshold
        valid_mask = areas > area_threshold
        filtered_boxes = boxes[valid_mask]
        filtered_labels = labels[valid_mask]
        
        return filtered_boxes, filtered_labels


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
