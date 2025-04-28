# -*- coding: utf-8 -*-
"""
@inproceedings{zhao2020maintaining,
  title={Maintaining discrimination and fairness in class incremental learning},
  author={Zhao, Bowen and Xiao, Xi and Gan, Guojun and Zhang, Bin and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR)},
  pages={13208--13217},
  year={2020}
}
https://arxiv.org/abs/1911.07053

Adapted from https://github.com/G-U-N/PyCIL/blob/master/models/wa.py, https://github.com/G-U-N/PyCIL/blob/master/utils/inc_net.py.
"""

import torch
from torch import nn
import copy
from torch.nn import functional as F
import numpy as np
from .finetune import Finetune


def KD_loss(pred, soft, T=2):
    '''
    Code Reference:
    https://github.com/G-U-N/PyCIL/blob/master/models/wa.py

    Compute the knowledge distillation loss.

    Args:
        pred (torch.Tensor): Predictions of the model.
        soft (torch.Tensor): Soft targets.
        T (float): Temperature parameter for softening the predictions. Default is 2.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    '''
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class IncrementalModel(nn.Module):
    '''
    Code Reference:
    https://github.com/G-U-N/PyCIL/blob/master/utils/inc_net.py
    
    A model consists with a backbone and a classifier.

    Args:
        backbone (nn.Module): Backbone network.
        feat_dim (int): Dimension of the extracted features.
        num_class (int): Number of classes in the dataset.
    '''
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = None
        
    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        '''
        Compute logits for the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Logits of the input data.
        '''
        logits = self.classifier(self.backbone(x)['features'])
        return logits

    def update_classifier(self, number_classes):
        '''
        Incrementally update the classifier with deepcopy.

        Args:
            number_classes (int): Number of classes after update.
        '''
        classifier = nn.Linear(self.feat_dim, number_classes)
        if self.classifier is not None:
            number_output = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            classifier.weight.data[:number_output] = weight
            classifier.bias.data[:number_output] = bias
        
        del self.classifier
        self.classifier = classifier

    def classifier_weight_align(self, incremental_number):
        '''
        Align the weight of the classifier after every task.

        Args:
            incremental_number (int): Number of classes added in the current task.
        '''
        weights = self.classifier.weight.data
        new_norm = torch.norm(weights[-incremental_number:, :], p=2, dim=1)
        old_norm = torch.norm(weights[:-incremental_number, :], p=2, dim=1)
        new_mean = torch.mean(new_norm)
        old_mean = torch.mean(old_norm)
        gamma = old_mean / new_mean
        self.classifier.weight.data[-incremental_number:, :] *= gamma

    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        logits = self.classifier(self.backbone(x)['features'])
        return logits
    
    def freeze(self):
        '''
        Freeze the model parameters.
        '''
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def extract_vector(self, x):
        '''
        Extract features from the backbone network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Extracted features.
        '''
        return self.backbone(x)["features"]


class WA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.network = IncrementalModel(self.backbone, feat_dim, kwargs['init_cls_num'])
        self.device = kwargs['device']
        self.old_network = None
        self.known_classes = 0
        self.total_classes = 0
        self.task_idx = 0
        # For buffer update
        self.total_classes_indexes = 0

    def observe(self, data):
        '''
        Do every current task.

        Args:
            data (dict): Dictionary containing input data and labels.

        Returns:
            tuple: Tuple containing predictions, accuracy, and loss.
        '''
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        self.network.to(self.device)
        if self.old_network:
            self.old_network.to(self.device)

        logits = self.network(x)
        loss = F.cross_entropy(logits, y)

        if self.task_idx > 0:
            kd_lambda = self.known_classes / self.total_classes
            loss_kd = KD_loss(
                logits[:, : self.known_classes],
                self.old_network(x),
            )
            loss = (1 - kd_lambda) * loss + kd_lambda * loss_kd


        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def inference(self, data):
        '''
        Perform inference on the input data.

        Args:
            data (dict): Dictionary containing input data and labels.

        Returns:
            tuple: Tuple containing predictions and accuracy.
        '''
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self.network(x)
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.network(x)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Do before every task for task initialization.

        Args:
            task_idx (int): Index of the current task.
            buffer (Buffer): Buffer object.
            train_loader (DataLoader): DataLoader for training data.
            test_loaders (list): List of DataLoaders for test data.
        '''
        self.total_classes += self.kwargs['init_cls_num']
        self.network.update_classifier(self.total_classes)

        self.total_classes_indexes = np.arange(self.known_classes, self.total_classes)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Do after every task for updating the model.

        Args:
            task_idx (int): Index of the current task.
            buffer (Buffer): Buffer object.
            train_loader (DataLoader): DataLoader for training data.
            test_loaders (list): List of DataLoaders for test data.
        '''
        if self.task_idx > 0:
            self.network.classifier_weight_align(self.total_classes - self.known_classes)
        self.old_network = copy.deepcopy(self.network).freeze()
        self.known_classes = self.total_classes

        # update buffer
        buffer.reduce_old_data(self.task_idx, self.total_classes)
        val_transform = test_loaders[0].dataset.trfms
        buffer.update(self.network, train_loader, val_transform,
                      self.task_idx, self.total_classes, self.total_classes_indexes,
                      self.device)

        self.task_idx += 1
