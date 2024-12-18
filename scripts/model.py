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
from torchvision.ops import nms, MultiScaleRoIAlign
import config
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.optim.swa_utils import SWALR
from functools import partial
from collections import defaultdict


class GCDDDetector(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001, trainable_backbone_layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.trainable_backbone_layers = trainable_backbone_layers
        self.detector = self._detector_setup(self.num_classes)
        self.loss_dict = defaultdict(lambda: 0.0)
        self.map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        self.map_alt = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)



    def _detector_setup(self, num_classes):
        
        detector = fasterrcnn_mobilenet_v3_large_fpn(
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            rpn_pre_nms_top_n_train=300,
            min_size=1024,
            max_size=2048,
        )

        # Get the input features for the classifier
        in_features = detector.roi_heads.box_predictor.cls_score.in_features

        # Replace the head with a new one (with the correct number of classes)
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    
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

        loss = loss_dict["loss_classifier"] + loss_dict["loss_box_reg"] + loss_dict["loss_objectness"] + loss_dict["loss_rpn_box_reg"]
        
        self.loss_dict["total_loss"] += loss.item()
        self.loss_dict["loss_classifier"] += loss_dict["loss_classifier"].item()
        self.loss_dict["loss_box_reg"] += loss_dict["loss_box_reg"].item()
        self.loss_dict["loss_objectness"] += loss_dict["loss_objectness"].item()
        self.loss_dict["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].item()
        self.loss_dict["step_count"] += 1
        
        return loss

    def validation_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
        self.map.update(preds, targets)

    def test_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
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
        preds = self.forward(images)
        return self._pred_to_df(image_names, preds)
        

    def on_train_epoch_end(self):
        step_count = self.loss_dict["step_count"]
        self.log_dict({
            "mean_total_loss" : self.loss_dict["total_loss"]/step_count,
            "mean_loss_classifier": self.loss_dict["loss_classifier"]/step_count,
            "mean_loss_box_reg": self.loss_dict["loss_box_reg"]/step_count,
            "mean_loss_objectness": self.loss_dict["loss_objectness"]/step_count,
            "mean_loss_rpn_box_reg": self.loss_dict["loss_rpn_box_reg"]/step_count,
        }, prog_bar=True)

        self.loss_dict.clear()

    def on_validation_epoch_end(self):
        map = self.map.compute()
        self.log("map_50", map["map_50"], prog_bar=True)


    def configure_optimizers(self):
        optimizer =  SGD(
            [{"params":self.parameters()}], lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        )

        # scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.0001, step_size_up=200, step_size_down=200)
        # scheduler = SWALR(optimizer, swa_lr=0.00001, anneal_epochs=20)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'epoch',
                "frequency": 1,
                "strict": True,
                "name": "cycle_scheduler",
                "monitor": "map_50"
            }

        }
