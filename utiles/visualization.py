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


def visualize(pred, image):  # для теста
    fig, ax = plt.subplots(1, 2, figsize=(15, 30))
    ax[1].imshow(pred)
    ax[2].imshow(image.permute(1, 2, 0))
    plt.show()


def visualize_masks(true, pred, image):  # для валидации
    fig, ax = plt.subplots(1, 3, figsize=(15, 45))
    ax[0].imshow(true)
    ax[1].imshow(pred)
    ax[2].imshow(image.permute(1, 2, 0))
    plt.show()


def my_loss(inputs, targets):
    inputs = torch.nn.functional.softmax(inputs)
    targets = torch.nn.functional.one_hot(targets).permute(0, 3, 1, 2)

    if targets.shape[1] == 1:
        one_hot_targets_class_2 = 1 - targets
        targets = torch.cat([targets, one_hot_targets_class_2], dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(
        inputs, targets, mode="multilabel", threshold=0.5
    )

    # then compute metrics with required reduction (see metric docs)
    # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
    f1_score_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
    f1_score_micro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    # f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")

    # print('f1', f1_score)
    print("f1_macro", f1_score_macro)
    print("f1_micro", f1_score_micro)
    print("recall", recall)
    print("precision", precision)
