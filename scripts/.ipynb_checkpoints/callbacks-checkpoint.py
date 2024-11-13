from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
import config
import os


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

    return callbacks
