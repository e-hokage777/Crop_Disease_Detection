## creating module
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim import SGD
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.ops import nms
import config
import pandas as pd


class GCDDDetector(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001, trainable_backbone_layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.trainable_backbone_layers = trainable_backbone_layers
        self.detector = self._detector_setup(self.num_classes)
        self.training_step_losses = []
        self.map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        self.map_alt = MeanAveragePrecision(iou_type="bbox", class_metrics=True)


    def _detector_setup(self, num_classes):
        anchor_generator = AnchorGenerator(sizes=config.ANCHOR_SIZES, aspect_ratios=config.ANCHOR_RATIOS)
        detector = fasterrcnn_mobilenet_v3_large_fpn(
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            rpn_nms_thresh=config.NMS_THRESH,
            box_nms_thresh=config.NMS_THRESH
        )

        # Get the input features for the classifier
        in_features = detector.roi_heads.box_predictor.cls_score.in_features

        # Replace the head with a new one (with the correct number of classes)
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        ## checking anchor generator
        detector.rpn.anchor_generator = anchor_generator

        return detector

    def _nms(self, preds):
        new_preds = []

        for pred in preds:
            keep_idxs = nms(pred["boxes"], pred["scores"], config.NMS_THRESH)
            new_pred = {}
            new_pred["boxes"] = pred["boxes"][keep_idxs]
            new_pred["labels"] = pred["labels"][keep_idxs]
            new_pred["scores"] = pred["scores"][keep_idxs]
            new_preds.append(new_pred)

        return new_preds


    def forward(self, x):
        if self.training:
            return self.detector(*x)
        return self.detector(x)

    def training_step(self, batch, batch_index):
        loss_dict = self.forward(batch)
        loss = sum(loss_dict.values())
        self.training_step_losses.append(loss)
        # self.log_dict(loss_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
        preds = self._nms(preds)
        self.map.update(preds, targets)

    def test_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
        preds = self._nms(preds)
        self.map_alt.update(preds, targets)

    def _pred_to_df(self, image_names, preds):
        items = []
        for name, pred in zip(image_names, preds):
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            for i in range(len(boxes)):
                current_box = boxes[i]
                items.append((name, labels[i], scores[i], current_box[0], current_box[1], current_box[2], current_box[3]))

        return pd.DataFrame(items, columns=["Image_ID", "class", "confidence", "xmin", "ymin", "xmax", "ymax"])
        
    def predict_step(self, batch, batch_index):
        image_names, images = batch
        preds = self._nms(self.forward(images))
        return self._pred_to_df(image_names, preds)
        

    def on_train_epoch_end(self):
        mean_loss = torch.stack(self.training_step_losses).mean()
        self.log("total_loss", mean_loss, prog_bar=True)
        self.training_step_losses = []

    def on_validation_epoch_end(self):
        map = self.map.compute()
        self.log("map_50", map["map_50"], prog_bar=True)

    # def on_test_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        params = [
            {"params": self.detector.backbone.parameters(), "lr":0.0001, "weight_decay":1e-5},
            {"params": self.detector.rpn.parameters(), "lr":0.001, "weight_decay":1e-4},
            {"params": self.detector.roi_heads.parameters(), "lr":0.001, "weight_decay":1e-4},
        ]
        return SGD(
            params, momentum=0.9
        )
