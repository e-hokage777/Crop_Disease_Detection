from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from torchvision.transforms import v2 as T
import torchvision.transforms.v2.functional as F
import torch
import torchvision
from torchvision.io import decode_image
from class_encoder import ClassEncoder

def get_num_classes(filepath, column):
    return pd.read_csv(filepath)[column].nunique() + 1

def get_labelencoder(filepath, column):
    # df = pd.read_csv(filepath)
    # label_encoder = LabelEncoder()
    # values = df[column].values.tolist()
    # label_encoder.fit(values)
    # return label_encoder
    return ClassEncoder(filepath, column, offset=1)

def get_file_dir():
    '''
    function to get the working directory of the currently calling file
    '''
    return os.path.dirname(os.path.abspath(__file__))


def get_transforms():
    # scale_transform = T.Lambda(lambda x: x/255.0, torchvision.tv_tensors._image.Image)
    return T.Compose([
        # T.RandomResizedCrop(256, scale=(0.8,1)),
        T.ToImage(),
        T.ColorJitter(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToDtype(torch.float32, scale=True),
    ])

    return t

def get_test_transforms():
    # scale_transform = T.Lambda(lambda x: x/255.0, torchvision.tv_tensors._image.Image)
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

def get_means_and_stds(annot_filepath, imgs_path, id_col = "Image_ID"):
    img_ids = pd.read_csv(annot_filepath)[id_col].unique()
    channel_sums = torch.zeros(3)
    channel_squared_sums = torch.zeros(3)
    num_pixels = 0
    for img_id in img_ids:
        image = decode_image(os.path.join(imgs_path, img_id))
        image = F.to_dtype(image, dtype=torch.float32, scale=True)
        num_pixels += image.numel()/3
        channel_sums += image.sum(axis=(1,2))
        channel_squared_sums += (image**2).sum(axis=(1,2))

    channel_means = channel_sums/num_pixels
    channel_stds = torch.sqrt(channel_squared_sums/num_pixels) - (channel_means**2)
    
    return channel_means, channel_stds
        
