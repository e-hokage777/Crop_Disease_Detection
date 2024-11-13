from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
import lightning as L
import config
import os

## callback for getting 
class ClassMAPDispCallback(L.Callback):
    def _map_per_class_dict(self, map_dict, encoder):
        holder = {}
        classes = self.encoder.inverse_transform(map["classes"].numpy())

        for i in range(len(classes)):
            holder[classes[i]] = map_dict["map_per_class"][i]

        return holder
        
    def on_test_epoch_end(self, trainer, pl_module):
        map = pl_module.map_alt.compute()
        map["classes_found"] = len(map["classes"])
        map_per_class_dict = self._map_per_class_dict(map, trainer.datamodule.label_encoder)
        del map["classes"]
        self.log_dict(map)
        self.log_dict(map_per_class_dict)


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
            filename="epoch-{epoch:02d}_lr="+str(config.LEARNING_RATE)+"_map@50={map_50:.2f}",
        )
        callbacks.append(checkpoint_callback)
        # callbacks.append(RichProgressBar())
        callbacks.append(ClassMAPDispCallback())

    return callbacks
