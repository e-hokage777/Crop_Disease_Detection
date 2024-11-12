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
import config


class GCDDDetector(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.detector = self._detector_setup(self.num_classes)
        self.training_step_losses = []
        self.map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

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
        self.map.update(preds, targets)

    def test_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
        self.map.update(preds, targets)

    def prediction_step(self, batch, batch_index):
        return self.forward(batch)

    def on_training_epoch_end(self):
        mean_loss = torch.stack(self.training_step_losses).mean()
        self.log("loss", mean_loss)

    def on_validation_epoch_end(self):
        map = self.map.compute()
        self.log("map_50", map["map_50"])

    def configure_optimizers(self):
        return SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        )
