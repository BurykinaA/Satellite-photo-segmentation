import torch
import numpy as np
import albumentations as albu
import torchvision
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from matplotlib.pyplot import imshow
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = None  # Отключает предупреждение

import warnings

warnings.filterwarnings("ignore")


weights = torch.tensor([1.0, 2.0])

CRITERIONS = {
    "ce": torch.nn.CrossEntropyLoss(),  # weight=torch.tensor([1.0, 3.0])
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "jaccar": JaccardDiceLoss(),
}


class ChaniseSatelliteDataset(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self.model = HRnet()
        self.criterion = CRITERIONS[config["criterion_type"]]
        self.iou = IoULoss()

        self.losss = []

    def get_dataset(self, learn=True):
        return SatelliteDataset(learn)

    def forward(self, x):
        predicted_heatmaps = self.model(x)
        return predicted_heatmaps

    def train_dataloader(self):
        dataset = self.get_dataset()
        params = self._config["dataset_params"]
        return torch.utils.data.DataLoader(
            dataset,
            params["batch_size"],
            num_workers=8,
            shuffle=False,
        )

    def test_dataloader(self):
        dataset = self.get_dataset(learn=False)
        params = self._config["dataset_params"]
        return torch.utils.data.DataLoader(
            dataset, batch_size=params["batch_size"], num_workers=8, shuffle=False
        )

    def val_dataloader(self):
        dataset = self.get_dataset(learn=False)
        params = self._config["dataset_params"]
        return torch.utils.data.DataLoader(
            dataset, batch_size=params["batch_size"], num_workers=8, shuffle=False
        )

    def configure_optimizers(self):
        params = self._config["optimizer_params"]
        optimizer = torch.optim.Adam(self.model.parameters(), **params)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predicted_heatmaps = self.model(images)
        loss = self.criterion(predicted_heatmaps, labels)
        if batch_idx % 100 == 0:
            print("train", loss)
        self.log("loss", loss)
        return {"loss": loss}

    def my_loss(self, inputs, targets):
        inputs = torch.nn.functional.softmax(inputs)
        targets = torch.nn.functional.one_hot(targets).permute(0, 3, 1, 2)

        if targets.shape[1] == 1:
            one_hot_targets_class_2 = 1 - targets
            targets = torch.cat([targets, one_hot_targets_class_2], dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            inputs, targets, mode="multilabel", threshold=0.5
        )
        self.losss.append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro"))

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predicted_heatmaps = self.model(images)
        loss = self.criterion(predicted_heatmaps, labels)
        print("test_loss", loss)
        self.log("test_loss", loss)

        self.my_loss(predicted_heatmaps, labels)
        print(sum(self.losss) / len(self.losss))

        predicted_masks = torch.argmax(predicted_heatmaps, dim=1)
        pred = predicted_masks.detach().cpu()[0]
        true = labels.detach().cpu()[0]
        image = images[0].detach().cpu()
        visualize_masks(true, pred, image)

        iou = self.iou(predicted_heatmaps, labels)
        # iou = compute_iou(predicted_masks.detach().cpu(), labels.detach().cpu())
        print("test_iou", iou)
        self.log("test_iou", iou)

        return {"test_loss": loss}
        # return {"test_loss": loss, "test_iou": iou, "specificity": specificity, "sensitivity": sensitivity,
        #       "score_auc": score_auc, "score_f1":score_f1, "score_acc":score_acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predicted_heatmaps = self.model(images)
        loss = self.criterion(predicted_heatmaps, labels)
        print(sum(self.losss) / len(self.losss) if len(self.losss) != 0 else 0)
        self.my_loss(predicted_heatmaps, labels)

        specificity, sensitivity, score_auc, score_f1, score_acc = calculate_metrics(
            labels, predicted_heatmaps
        )
        #         print({"specificity": specificity, "sensitivity": sensitivity,
        #               "score_auc": score_auc, "score_f1":score_f1, "score_acc":score_acc})

        #         print("val_loss", loss)

        self.log("val_loss", loss)
        # predicted_masks = torch.argmax(predicted_heatmaps, dim=1)
        iou = self.iou(predicted_heatmaps, labels)
        # print("val_iou", iou)
        self.log("val_iou", iou)
        return {"val_loss": loss, "val_iou": iou}

    def on_epoch_end(self):
        self.losss = []
        # Сохранение весов модели после каждой эпохи в указанную папку
        save_path = os.path.join(
            "/home/jupyter/datasphere/project/weights",
            f"model_china_{self.current_epoch}.pth",
        )
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":

    config = {
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
        filename="model-china-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    module = ChaniseSatelliteDataset(config)
    model.model.load_state_dict(torch.load(saved_weights_path))

    # model2 = ChaniseSatelliteDataset.load_from_checkpoint(checkpoint_path='/home/jupyter/datasphere/project/weights/model-china-epoch=01-val_loss=0.21.ckpt')

    # logger = pl.loggers.TensorBoardLogger("./logs", name='dice')
    trainer2 = pl.Trainer(
        accelerator="gpu",  # "gpu",
        # logger=logger,
        log_every_n_steps=10,
        max_epochs=3,
        default_root_dir="/home/jupyter/datasphere/project/",
        callbacks=[checkpoint_callback],
    )
    traine2r.fit(model2)
    trainer.test()
