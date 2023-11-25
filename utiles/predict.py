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


def get_final_prediction(picture, net=1):
    test_dataloader = torch.utils.data.DataLoader(TestDataset(picture), batch_size=8)
    if net == 1:
        predictions = trainer1.predict(model1, dataloaders=test_dataloader)
    elif net == 2:
        predictions = trainer2.predict(model2, dataloaders=test_dataloader)
    else:
        predictions = trainer3.predict(model3, dataloaders=test_dataloader)

    for i in range(len(predictions)):
        predicted_masks = torch.argmax(predictions[i], dim=1)
        pred = predicted_masks.detach().cpu()
        predictions[i] = pred

    all_images = []
    to_pil = transforms.ToPILImage()

    for tensor in predictions:
        reshaped_tensor = tensor.view(-1, 512, 512).float()
        all_images.extend(reshaped_tensor)

    for i in range(len(all_images)):
        all_images[i] = all_images[i] / 255.0  # не видно но получаем предикт
        all_images[i] = to_pil(all_images[i])

    def reconstruct_mask(
        mask_patches,
        image_size=Image.open(picture).convert("RGB").size,
        patch_size=(512, 512),
    ):
        reconstructed_mask = Image.new("L", image_size)
        index = 0

        for i in range(0, image_size[0], patch_size[0]):
            for j in range(0, image_size[1], patch_size[1]):
                patch_mask = mask_patches[index]
                reconstructed_mask.paste(patch_mask, (i, j))
                index += 1

        return reconstructed_mask

    tmp = reconstruct_mask(all_images)
    plt.imshow(tmp)
    if net == 1 or net == 3:
        filename = os.path.basename(picture)
    else:
        filename = (
            os.path.basename(picture)[:-4] + "_1" + os.path.basename(picture)[-4:]
        )
    path_to_save = os.path.join(
        "/home/jupyter/datasphere/project/answers/test_new", filename
    )
    tmp.save(path_to_save)
