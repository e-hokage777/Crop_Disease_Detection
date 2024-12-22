from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from torchvision.transforms import v2 as T
import torchvision.transforms.v2.functional as F
import torch
import torchvision
from torchvision.io import decode_image
from class_encoder import ClassEncoder
from torchvision.io import read_image

def get_num_classes(filepath, column):
    return pd.read_csv(filepath)[column].nunique() + 1

def get_labelencoder(filepath, column):
    return ClassEncoder(filepath, column, offset=1)

def get_file_dir():
    '''
    function to get the working directory of the currently calling file
    '''
    return os.path.dirname(os.path.abspath(__file__))


def get_transforms():
    return T.Compose([
        T.ToImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ClampBoundingBoxes(),
        T.SanitizeBoundingBoxes(),
        T.ToDtype(torch.float32, scale=True),
    ])


    return t

def get_test_transforms():
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


## function to remove outliers
def remove_outliers(df):
    width = df["xmax"] - df["xmin"]
    height = df["ymax"] - df["ymin"]
    aspect = width/height

    ## handling outliers
    min_dim = 1
    max_dim = 700
    min_ratio = 0.5
    max_ratio = 2
    df = df[((width >= min_dim) & (width <= max_dim) &
             (height >= min_dim) & (height <= max_dim) &
             (aspect >= min_ratio) & (aspect <= max_ratio))]
    
    return df

## function to remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates(subset=["xmin", "xmax", "ymin", "ymax"])




## function to clean data (CLEANING SHOULD BE DONE ONLY ON TRAIN DATA)
def clean_bbox_data(df, remove_dups=False, remove_outs=False):
    if remove_dups:
        df = remove_duplicates(df)
    if remove_outs:
        df = remove_outliers(df)
    
    return df


def predict(model, img_paths, inference_transforms, encoder):
    model.eval() ## prepping model for prediction
    images = []
    for img_path in img_paths:
        image = read_image(img_path)
        image = inference_transforms(image)
        images.append(image)
        
    with torch.no_grad():
        preds = model(images)

    results = []
    for path, pred in zip(img_paths, preds):
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        labels = encoder.inverse_transform(labels)

        results.append({"image": path,"boxes": boxes, "labels": labels, "scores": scores})

    return results