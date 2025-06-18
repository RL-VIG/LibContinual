# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/0002ZL0SRSPDP22,
  author       = {Zifeng Wang and
                  Zizhao Zhang and
                  Chen{-}Yu Lee and
                  Han Zhang and
                  Ruoxi Sun and
                  Xiaoqi Ren and
                  Guolong Su and
                  Vincent Perot and
                  Jennifer G. Dy and
                  Tomas Pfister},
  title        = {Learning to Prompt for Continual Learning},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2022, New Orleans, LA, USA, June 18-24, 2022},
  pages        = {139--149},
  publisher    = {{IEEE}},
  year         = {2022}
}

https://arxiv.org/abs/2112.08654

Adapted from https://github.com/GT-RIPL/CODA-Prompt
"""

import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.resnet import *

class Model(nn.Module):
    def __init__(self, backbone, embed_dim, total_cls_num):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, total_cls_num, bias=True)

    def forward(self, x, train=True):
        feat, reduce_sim = self.backbone(x, train=train)
        return self.classifier(feat), reduce_sim

class L2P(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.total_cls_num = kwargs['num_class']
        self.task_num = kwargs['task_num']
        self.embed_dim = kwargs['feat_dim']
        self.pull_constraint_coeff = kwargs['pull_constraint_coeff']
        self.cur_task_id = 0
        self._known_classes = 0
        
        self.network = Model(backbone, self.embed_dim, self.total_cls_num)        
        self.network.backbone.create_prompt(
            prompt_flag = 'l2p', 
            length = kwargs['prompt_length'], # L_p
            prompt_init = nn.init.uniform_,
            pool_size = kwargs['pool_size'],  # M
            top_k = kwargs['top_k'],          # N
            num_layers = 1,
            embed_dim = self.embed_dim
        )
        self.network.to(self.device)

        self.unfrezeed_params = []
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
            if 'prompt' in name or 'classifier' in name:
                param.requires_grad_(True)
                self.unfrezeed_params.append(param)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self.cur_task_id = task_idx

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logits, reduce_sim = self.network(x, train=True)

        if self.cur_task_id == 0:
            mask = np.arange(self.init_cls_num)
        else:
            mask = np.arange(self.inc_cls_num) + self._known_classes

        not_mask = np.setdiff1d(np.arange(self.total_cls_num), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = F.cross_entropy(logits, y) - self.pull_constraint_coeff * reduce_sim      

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unfrezeed_params, 1.0)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item() / x.size(0)

        return pred, acc, loss

    def inference(self, data):
        
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logits, _ = self.network(x, train=False)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item() / x.size(0)
        return pred, acc

    def get_parameters(self, config):

        return self.unfrezeed_params
