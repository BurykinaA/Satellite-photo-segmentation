from models_zoo.basic_models import *

config = {
    "model_type": "pretrain-deeplab",  # "pretrain-deeplab", "pretrain-unet"
    "model_params": {
        "classes": 2,
        "encoder_name": "resnet152",
        # "encoder_weights": "imagenet",
        "activation": None,
    },
    "criterion_type": "ce",
    "optimizer_params": {"lr": 0.0001},
    "dataset_params": {
        "batch_size": 8,
    },
}


# Определите колбэк ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="/home/jupyter/datasphere/project/weights",
    filename="model-weight-dlv3p-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)


# module = PLSatelliteDataset(config)
# model.model.load_state_dict(torch.load(saved_weights_path))

model3 = PLSatelliteDataset.load_from_checkpoint(
    checkpoint_path="/home/jupyter/datasphere/project/weights/model-weight-dlv3p-epoch=02-val_loss=0.25.ckpt"
)

# logger = pl.loggers.TensorBoardLogger("./logs", name='dice')
trainer3 = pl.Trainer(
    accelerator="gpu",  # "gpu",
    # logger=logger,
    log_every_n_steps=10,
    max_epochs=2,
    default_root_dir="/home/jupyter/datasphere/project/",
    callbacks=[checkpoint_callback],
)
# trainer1.fit(model1)
# trainer1.test()
