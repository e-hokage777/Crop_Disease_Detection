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
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torch.optim.swa_utils import SWALR


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

        print("INITIAL LEARNING_RATE:", self.learning_rate)


    def _detector_setup(self, num_classes):
        detector = fasterrcnn_mobilenet_v3_large_fpn(
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        )

        # Get the input features for the classifier
        in_features = detector.roi_heads.box_predictor.cls_score.in_features

        # Replace the head with a new one (with the correct number of classes)
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        # anchor_generator = AnchorGenerator(sizes=config.ANCHOR_SIZES, aspect_ratios=config.ANCHOR_RATIOS)
        # backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # # backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "pool"], output_size=14, sampling_ratio=2)
        # backbone = _mobilenet_extractor(backbone, True, 3)

        # detector = FasterRCNN(
        #     backbone,
        #     num_classes=self.num_classes,
        #     # image_mean=(0.4763, 0.5514, 0.3163),
        #     # image_std=(0.2901, 0.2821, 0.2724),
        #     rpn_anchor_generator = anchor_generator,
        #     box_roi_pool = box_roi_pool,
        #     # box_nms_thresh=0.3
        # )

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
        # preds = self._nms(preds)
        self.map.update(preds, targets)

    def test_step(self, batch, batch_index):
        images, targets = batch
        preds = self.forward(images)
        # preds = self._nms(preds)
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
        # preds = self._nms(self.forward(images))
        preds = self.forward(images)
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
        # params = [
        #     {"params": self.detector.backbone.parameters(), "lr":0.0001, "weight_decay":1e-5},
        #     {"params": self.detector.rpn.parameters(), "lr":0.001, "weight_decay":1e-4},
        #     {"params": self.detector.roi_heads.parameters(), "lr":0.001, "weight_decay":1e-4},
        # ]
        optimizer =  SGD(
            [{"params":self.parameters()}], lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        )
        
        # scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=0)
        # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.08, step_size_up=200, step_size_down=200)
        scheduler = SWALR(optimizer, swa_lr=0.001, anneal_epochs=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
                "strict": True,
                "name": "cycle_scheduler",
            }
        }
