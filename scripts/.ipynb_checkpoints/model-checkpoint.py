## creating module
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim import SGD
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms, nms
import config


class GCDDDetector(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.detector = self._detector_setup(self.num_classes)
        self.training_step_losses = []
        self.map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        self.map_alt = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        ## saving in logs
        self.log("lr", self.learning_rate)

    def _detector_setup(self, num_classes):
        ## getting backbone weights
        backbone_weights = torch.load(
            config.BACKBONE_PATH,
            weights_only=True,
            map_location="cpu"
        )

        detector = fasterrcnn_mobilenet_v3_large_fpn(
            FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
            weights_backbone=backbone_weights,
        )
        # Get the input features for the classifier
        in_features = detector.roi_heads.box_predictor.cls_score.in_features

        # Replace the head with a new one (with the correct number of classes)
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return detector

    ## function to filter with non-max suppression
    def _nms(self, preds):
        if not type(preds) == list:
            preds = [preds]

        new_preds = []
        
        for pred in preds:
            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]

            filtered_idxs = nms(boxes, scores, config.NMS_THRESH)

            new_pred = dict()
            new_pred["boxes"] = boxes[[filtered_idxs]]
            new_pred["labels"] = labels[[filtered_idxs]]
            new_pred["scores"] = scores[[filtered_idxs]]
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

    def prediction_step(self, batch, batch_index):
        return self.forward(batch)

    def on_training_epoch_end(self):
        mean_loss = torch.stack(self.training_step_losses).mean()
        self.log("loss", mean_loss, prog_bar=True)
        self.training_step_losses = []

    def on_validation_epoch_end(self):
        map = self.map.compute()
        self.log("map_50", map["map_50"], prog_bar=True)

    # def on_test_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        return SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        )
