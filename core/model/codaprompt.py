# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SmithKGCKAPFK23,
  author       = {James Seale Smith and
                  Leonid Karlinsky and
                  Vyshnavi Gutta and
                  Paola Cascante{-}Bonilla and
                  Donghyun Kim and
                  Assaf Arbelle and
                  Rameswar Panda and
                  Rog{\'{e}}rio Feris and
                  Zsolt Kira},
  title        = {CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free
                  Continual Learning},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2023, Vancouver, BC, Canada, June 17-24, 2023},
  pages        = {11909--11919},
  publisher    = {{IEEE}},
  year         = {2023}
}

https://arxiv.org/abs/2211.13218

Adapted from https://github.com/GT-RIPL/CODA-Prompt
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


class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        
    def forward(self, x, train=True):
        if train:
            feat, loss = self.backbone(x, train=True)
            return self.classifier(feat), loss
        else:
            feat = self.backbone(x, train=False)
            return self.classifier(feat)


class CodaPrompt(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = Model(self.backbone, feat_dim, kwargs['init_cls_num'])
        self.network.backbone.create_prompt('coda', n_tasks = kwargs['task_num'], prompt_param=[kwargs['pool_size'], kwargs['prompt_length'], kwargs['mu']])
        self.task_idx = 0
        self.kwargs = kwargs
        
        self.last_out_dim = 0

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.network.backbone.task_id = task_idx
        
        in_features = self.network.classifier.in_features
        out_features = self.network.classifier.out_features
        new_out_features = self.kwargs['init_cls_num'] + task_idx * self.kwargs['inc_cls_num']
        new_fc = nn.Linear(in_features, new_out_features)
        new_fc.weight.data[:out_features] = self.network.classifier.weight.data
        new_fc.bias.data[:out_features] = self.network.classifier.bias.data
        self.network.classifier = new_fc
        self.network.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        self.out_dim = new_out_features
        self.dw_k = torch.tensor(np.ones(self.out_dim + 1, dtype=np.float32)).to(self.device)

    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit, loss = self.network(x, train=True)

        logit[:,:self.last_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]

        loss += (self.loss_fn(logit, y) * dw_cls).mean()
        
        pred = torch.argmax(logit, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss
    
        

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self.last_out_dim = self.out_dim

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logit = self.network(x, train=False)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)


    def get_parameters(self, config):
        return list(self.network.backbone.prompt.parameters()) + list(self.network.classifier.parameters())
