from lightning.pytorch.callbacks import (ModelCheckpoint,
EarlyStopping,
RichProgressBar, BasePredictionWriter, StochasticWeightAveraging, LearningRateMonitor)
import lightning as L
import config
import os
import pandas as pd

## callback for getting map stats
class ClassMAPDispCallback(L.Callback):
    def __init__(self, save_path=None):
        self.save_path = save_path
    def _map_per_class_dict(self, map_dict, encoder):
        holder = {}
        classes = encoder.inverse_transform(map_dict["classes"].numpy())

        for i in range(len(classes)):
            holder[classes[i]] = map_dict["map_per_class"][i].item()

        return holder
        
    def on_test_epoch_end(self, trainer, pl_module):
        map = pl_module.map_alt.compute()
        map_per_class_dict = self._map_per_class_dict(map, trainer.datamodule.label_encoder)
        self.log_dict({
            "map": map["map"],
            "map_small": map["map_small"],
            "map_large": map["map_large"],
            "map_50": map["map_50"],
            "map_75": map["map_75"],
            "mar_10": map["mar_10"],
            "mar_100": map["mar_100"],
            "num_classes": len(map["classes"])
        })
        self.log_dict(map_per_class_dict)

        if self.save_path:
            pd.Series(map_per_class_dict, name="map").to_csv(self.save_path)
            

        

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        full_df = pd.concat(predictions, axis=0)
        full_df["class"] = trainer.datamodule.label_encoder.inverse_transform(full_df["class"])
        sub_index = 0
        existing_subs = sorted(os.listdir(self.output_dir))
        sub_indices = [int(item.split("_")[-1].split(".")[0])  for item in os.listdir(self.output_dir) if item.startswith("submission")]
        if sub_indices:
            sub_index = max(sub_indices)+1
        full_df.to_csv(os.path.join(self.output_dir, f"submission_{sub_index}.csv"), index=False)

class MetricsLogCallback(L.Callback):
    def __init__(self, metrics, logging_interval="step"):
        self.metrics = metrics
        self.logging_interval = logging_interval

    def _common_step(self, trainer, pl_module):
        metrics_dict = {}
        for metric in self.metrics:
            value = trainer.callback_metrics.get(metric)
            if value:
                metrics_dict[metric] = value

        if metrics_dict:
            pl_module.log_dict(metrics_dict, prog_bar=True, logger=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.logging_interval != "step":
            self._common_step(trainer, pl_module)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.logging_interval == "step":
            self._common_step(trainer, pl_module)



def get_callbacks(checkpoint_monitor="map_50"):
    checkpoint_savepath = (
        os.environ.get("GCDD_CHECKPOINT_SAVE_PATH") or config.CHECKPOINT_SAVEPATH
    )
    callbacks = []
    if checkpoint_monitor:
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_monitor,
            dirpath=checkpoint_savepath,
            mode="max",
            every_n_epochs=1,
            save_top_k=1,
            save_on_train_epoch_end=False,
            filename="epoch-{epoch:02d}_map@50={map_50:.2f}",
        )
        callbacks.append(checkpoint_callback)
        
    callbacks.append(RichProgressBar())
    callbacks.append(ClassMAPDispCallback(save_path=config.METRICS_SAVE_PATH))
    callbacks.append(PredictionWriter(config.SUBMISSION_PATH, write_interval="epoch"))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(MetricsLogCallback(metrics=["cycle_scheduler"]))

    return callbacks
