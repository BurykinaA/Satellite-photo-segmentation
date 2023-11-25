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
        learn=True,
        root_img="/home/jupyter/datasphere/project/data/img",
        root_mask="/home/jupyter/datasphere/project/data/mask",
        patch_size=(512, 512),
    ):
        self.image_dir = root_img
        self.mask_dir = root_mask

        self.image_names = sorted(i for i in os.listdir(root_img) if "png" in i)
        self.mask_names = sorted(i for i in os.listdir(root_mask) if "png" in i)
        if learn:
            self.image_names = self.image_names[:-1]
            self.mask_names = self.mask_names[:-1]
        else:
            self.image_names = self.image_names[-1:]
            self.mask_names = self.mask_names[-1:]

        self.patch_size = patch_size

        self.aug = get_training_transform()
        self.post_transform = torchvision.transforms.ToTensor()

        self.cur_image = -1
        self.cur_image_patches = None
        self.cur_mask_patches = None

        self.patches_count = [0 for i in range(len(self.image_names))]
        for i in range(len(self.image_names)):
            image = Image.open(
                os.path.join(self.image_dir, self.image_names[i])
            ).convert("RGB")
            a = (image.size[0] + self.patch_size[0] - 1) // self.patch_size[0]
            b = (image.size[1] + self.patch_size[1] - 1) // self.patch_size[1]
            self.patches_count[i] = a * b

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

    def _split_image_into_patches(self, image, mask, patch_size=(3000, 3000)):
        # image в формате тензора, mask в формате Image
        image_patches = []
        mask_patches = []

        # to_pil = transforms.ToPILImage()
        # image = to_pil(image)

        for i in range(0, image.width, patch_size[0]):
            for j in range(0, image.height, patch_size[1]):
                patch_image = image.crop((i, j, i + patch_size[0], j + patch_size[1]))
                patch_mask = mask.crop((i, j, i + patch_size[0], j + patch_size[1]))

                image_patches.append(patch_image)
                mask_patches.append(patch_mask)

        # возвращает списки патчей в формате Image
        return image_patches, mask_patches

    def __getitem__(self, item):
        general_item = 0
        if item >= len(self):
            raise IndexError
        while item >= sum(self.patches_count[: general_item + 1]):
            general_item += 1

        if general_item != self.cur_image:
            image = Image.open(
                os.path.join(self.image_dir, self.image_names[general_item])
            ).convert("RGB")
            mask = Image.open(
                os.path.join(self.mask_dir, self.mask_names[general_item])
            ).convert("L")

            self.cur_image = general_item
            (
                self.cur_image_patches,
                self.cur_mask_patches,
            ) = self._split_image_into_patches(image, mask, self.patch_size)

        image_patch = np.array(
            self.cur_image_patches[item - sum(self.patches_count[:general_item])]
        )
        mask_patch = np.array(
            self.cur_mask_patches[item - sum(self.patches_count[:general_item])]
        )

        transformed = self.aug(image=image_patch, mask=mask_patch)
        image_patch, mask_patch = transformed["image"], transformed["mask"]
        image_patch = self.post_transform(image_patch)

        return image_patch, torch.tensor(mask_patch).long()

    def __len__(self):
        # return len(self.image_names)
        return sum(self.patches_count)
