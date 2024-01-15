import torch
from torch import nn
import copy
from torch.nn import functional as F
import math
import numpy as np
from .finetune import Finetune

def KD_loss(pred, soft, T=2):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class WA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        # 用来初始化各自算法需要的对象
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.backbone = backbone
        self.device = kwargs['device']
        self.old_backbone = None
        self.known_classes = 0
        self.total_classes = 0
        self.task_idx = 0
        # For buffer update
        self.total_classes_indexes = 0

    def observe(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        self.backbone.to(self.device)
        if self.old_backbone:
            self.old_backbone.to(self.device)

        logits = self.backbone(x)["logits"]
        loss = F.cross_entropy(logits, y)

        if self.task_idx > 0:
            kd_lambda = self.known_classes / self.total_classes
            loss_kd = KD_loss(
                logits[:, : self.known_classes],
                self.old_backbone(x)["logits"],
            )
            loss = (1 - kd_lambda) * loss + kd_lambda * loss_kd


        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self.backbone(x)["logits"]
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.backbone(x)["logits"]

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.total_classes = buffer.total_classes
        self.backbone.update_classifier(self.total_classes)

        self.total_classes_indexes = np.arange(self.known_classes, self.total_classes)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.backbone.weight_align(self.total_classes - self.known_classes)
        self.old_backbone = self.backbone.copy().freeze()
        self.known_classes = self.total_classes

        # update buffer
        buffer.reduce_old_data(self.task_idx, self.total_classes)
        val_transform = test_loaders[0].dataset.trfms
        buffer.update(self.backbone, train_loader, val_transform,
                      self.task_idx, self.total_classes, self.total_classes_indexes,
                      self.device)

        self.task_idx += 1


