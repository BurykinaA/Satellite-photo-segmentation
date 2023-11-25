MODELS = {
    "pretrain-unet": smp.Unet,
    "pretrain-deeplab": smp.DeepLabV3Plus,
    "pretrain-fpn": smp.FPN,
    "pretrain-psp": smp.PSPNet,
}

CRITERIONS = {
    "ce": torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0])),  #
    "dice": DiceLoss(),
    "iou": IoULoss(),
}


class PLSatelliteDataset(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self.model = MODELS[config["model_type"]](**config["model_params"])
        self.criterion = CRITERIONS[config["criterion_type"]]
        self.iou = IoULoss()

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

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predicted_heatmaps = self.model(images)
        loss = self.criterion(predicted_heatmaps, labels)
        print("test_loss", loss)
        self.log("test_loss", loss)

        # my_loss(predicted_heatmaps, labels)

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

        # my_loss(predicted_heatmaps, labels)

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
        # Сохранение весов модели после каждой эпохи в указанную папку
        save_path = os.path.join(
            "/home/jupyter/datasphere/project/weights",
            f"model_epoch_{self.current_epoch}.pth",
        )
        torch.save(self.model.state_dict(), save_path)
