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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from operator import add


def calculate_metrics(y_true, y_pred):
    try:
        y_true = y_true.cpu().numpy()
        y_true = y_true > 0.5
        y_true = y_true.astype(np.uint8)
        y_true = y_true.reshape(-1)

        y_pred = y_pred.cpu().numpy()
        y_pred_auc = y_pred.reshape(-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.reshape(-1)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        score_auc = roc_auc_score(y_true, y_pred_auc)
        score_f1 = 2 * tp / (2 * tp + fn + fp)
        score_acc = (tp + tn) / (tp + tn + fn + fp)

        return [specificity, sensitivity, score_auc, score_f1, score_acc]
    except:
        return [0, 0, 0, 0, 0]


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.nn.functional.softmax(inputs)
        targets = torch.nn.functional.one_hot(targets).permute(0, 3, 1, 2)

        if targets.shape[1] == 1:
            one_hot_targets_class_2 = 1 - targets
            targets = torch.cat([targets, one_hot_targets_class_2], dim=1)

        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)

        inter = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2 * inter) / (union + self.eps)
        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.nn.functional.softmax(inputs)
        targets = torch.nn.functional.one_hot(targets).permute(0, 3, 1, 2)

        if targets.shape[1] == 1:
            one_hot_targets_class_2 = 1 - targets
            targets = torch.cat([targets, one_hot_targets_class_2], dim=1)

        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return bce_loss + dice_loss


class JaccardBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard = IoULoss()

    def forward(self, inputs, targets):
        jaccard_loss = self.jaccard(inputs, targets)
        bce_loss = self.bce(inputs, targets)

        return bce_loss + jaccard_loss


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2.0, smooth=1):
        inputs = torch.nn.functional.softmax(inputs)
        targets = torch.nn.functional.one_hot(targets).permute(0, 3, 1, 2)
        BCE = nn.functional.binary_cross_entropy(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return focal_loss


class JaccardDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.jaccard = IoULoss()

    def forward(self, inputs, targets):
        jaccard_loss = self.jaccard(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return dice_loss + jaccard_loss
