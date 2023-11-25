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


def print_getitem(b):
    fig, axes = plt.subplots(1, 2, figsize=(50, 50))
    axes[0].imshow(b[0].permute(1, 2, 0))
    axes[1].imshow(b[1].squeeze(0), cmap="gray")


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        # albu.RandomContrast(limit=0.2, p=0.25),
    ]
    return albu.Compose(train_transform)


class SatelliteDataset(Dataset):
    def __init__(
        self,
        root_img="/home/jupyter/datasphere/project/data/img",
        root_mask="/home/jupyter/datasphere/project/data/mask",
    ):
        self.image_dir = root_img
        self.mask_dir = root_mask
        self.image_names = sorted(i for i in os.listdir(root_img) if "png" in i)
        self.mask_names = sorted(i for i in os.listdir(root_mask) if "png" in i)

        self.aug = get_training_transform()
        self.post_transform = torchvision.transforms.ToTensor()
        self.post_transform = torchvision.transforms.ToTensor()

    def print_mask(self, item):
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[item]))
        mask.show()

    def print_img(self, item):
        image = Image.open(os.path.join(self.image_dir, self.image_names[item]))
        image.show()

    def print_img_mask(self, item):
        image = Image.open(os.path.join(self.image_dir, self.image_names[item]))
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[item]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        plt.show()

    def print_mask_on_img(self, item):
        image_path = os.path.join(self.image_dir, self.image_names[item])
        mask_path = os.path.join(self.mask_dir, self.mask_names[item])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(image.size, Image.NEAREST)
        mask_np = np.array(mask)
        cmap = plt.cm.get_cmap("viridis", 2)

        plt.figure(figsize=(10, 5))
        plt.imshow(image)
        plt.imshow(mask_np, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        plt.title("Image with Overlayed Mask")
        plt.axis("off")
        plt.show()

    def __getitem__(self, item):
        image = Image.open(
            os.path.join(self.image_dir, self.image_names[item])
        ).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[item])).convert(
            "L"
        )
        image = np.array(image)
        mask = np.array(mask)

        transformed = self.aug(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        image = self.post_transform(image)

        return image, torch.tensor(mask).long()

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    df = SatelliteDataset()
