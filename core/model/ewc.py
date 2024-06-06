# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/KirkpatrickPRVD16,
  author       = {James Kirkpatrick and
                  Razvan Pascanu and
                  Neil C. Rabinowitz and
                  Joel Veness and
                  Guillaume Desjardins and
                  Andrei A. Rusu and
                  Kieran Milan and
                  John Quan and
                  Tiago Ramalho and
                  Agnieszka Grabska{-}Barwinska and
                  Demis Hassabis and
                  Claudia Clopath and
                  Dharshan Kumaran and
                  Raia Hadsell},
  title        = {Overcoming catastrophic forgetting in neural networks},
  journal      = {CoRR},
  volume       = {abs/1612.00796},
  year         = {2016}
}

https://arxiv.org/abs/1612.00796

Adapted from https://github.com/G-U-N/PyCIL/blob/master/models/ewc.py
"""


import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune
from core.model.backbone.resnet import *
import numpy as np
from torch.utils.data import DataLoader
from torch import optim


class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        
    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        logits = self.classifier(self.backbone(x)['features'])
        return logits

class EWC(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = Model(self.backbone, feat_dim, kwargs['init_cls_num'])
        self.fisher = None
        self.lamda = self.kwargs['lamda']
        
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        in_features = self.network.classifier.in_features
        out_features = self.network.classifier.out_features
        
        new_fc = nn.Linear(in_features, self.kwargs['init_cls_num'] + task_idx * self.kwargs['inc_cls_num'])
        new_fc.weight.data[:out_features] = self.network.classifier.weight.data
        new_fc.bias.data[:out_features] = self.network.classifier.bias.data
        self.network.classifier = new_fc
        self.network.to(self.device)


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.network(x)
        if self.task_idx == 0:
            loss = F.cross_entropy(logit, y)
        else:
            old_classes = self.network.classifier.out_features - self.kwargs['inc_cls_num']
            loss = F.cross_entropy(logit[:, old_classes:], y - old_classes)
            loss += self.lamda * self.compute_ewc()
            
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss


    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx == 0:
            self.fisher = self.getFisher(train_loader)
        else:
            """
            Code Reference:
            https://github.com/G-U-N/PyCIL/blob/master/models/ewc.py
            """
            alpha = 1 - self.kwargs['inc_cls_num']/self.network.classifier.out_features
            new_finsher = self.getFisher(train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                    alpha * self.fisher[n]
                    + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
            
        self.mean = {
            n: p.clone().detach()
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        
        for n in self.mean.keys():
            self.mean[n].to(self.device)
        
    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logit = self.network(x)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)
    

    def getFisher(self, train_loader):
        """
        Code Reference:
        https://github.com/G-U-N/PyCIL/blob/master/models/ewc.py
        """
        fisher = {
            n: torch.zeros(p.shape).to(self.device)
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        self.network.train()
        optimizer = optim.SGD(self.network.parameters(), lr=0.1)
        for batch_idx, data in enumerate(train_loader):
            x, y = data['image'], data['label']
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits = self.network(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self.network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(0.0001))
        return fisher

    def compute_ewc(self):
        """
        Code Reference:
        https://github.com/G-U-N/PyCIL/blob/master/models/ewc.py
        """
        loss = 0
        for n, p in self.network.named_parameters():
            if n in self.fisher.keys():
                loss += (
                    torch.sum(
                        (self.fisher[n])
                        * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                    )
                    / 2
                )
        return loss
    
    def get_parameters(self,  config):
        train_parameters = []
        train_parameters.append({"params": self.network.parameters()})
        return train_parameters