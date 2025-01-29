# -*- coding: utf-8 -*-
"""
TODO: citation

Adapted from TODO: source
"""

import math
import torch
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from .backbone.transformer import ResidualAttentionBlock
from .backbone.clip import tokenize, CLIP
from .backbone.vit import ViTZoo

VIT = ViTZoo
CLIP = CLIP

class DMNSP(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.label_smoothing = kwargs['label_smoothing']

        self._cur_task_id = -1
        self._known_classes = 0
        self.visual_U = []
        self.lamda = [[0 for _ in range(12)] for _ in range(12)]
        self.lamda_scale = kwargs['lamda_scale']

        self.accm_class_names = []   
        self.curr_class_names = []
        self.accm_text_tokens = None
        self.curr_text_tokens = None

        self.prompt_template = kwargs['prompt_template']
        
        self._network = backbone

        for name, param in self._network.named_parameters():
            if 'adapt' not in name:
                param.requires_grad = False

        if isinstance(self._network, VIT):
            self.visual_transformer_blocks = [module for module in self._network.modules() if isinstance(module, ResidualAttentionBlock)]

            self.classifier_pool = nn.ModuleList([
                nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
                [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)]
            )

        elif isinstance(self._network, CLIP):
            self.visual_transformer_blocks = [module for name, module in self._network.named_modules() if isinstance(module, ResidualAttentionBlock) and 'visual' in name]
        else:
            assert 0

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        if isinstance(self._network, CLIP):
            features_img, features_txt, logits_per_img, logits_per_txt = self._network(x, self.curr_text_tokens)
        elif isinstance(self._network, ViTZoo):
            features = self._network(x)
            logits_per_img = []
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits_per_img.append(prompts(features))
            logits_per_img = torch.cat(logits_per_img, dim=1)

        loss = F.cross_entropy(logits_per_img, y, label_smoothing=self.label_smoothing)

        preds = logits_per_img.softmax(dim=-1).argmax(dim=1)
        acc = preds.eq(y).sum().item() / y.size(0)

        loss.backward()

        if self._cur_task_id > 0:

            if isinstance(self._network, VIT):

                for name, param in self._network.named_parameters():
                    for i in range(12):
                        if 'adapt' in name and 'down' in name and 'weight' in name:

                            v = self.visual_U[i].to(self.device)
                            v_ = torch.mm(param.grad.data, v)
                            param.grad.data = torch.mm(v_, v.T) * self.lamda[int(name.split(".")[3])][i]

                        elif 'adapt' in name and 'up' in name and 'weight' in name:

                            v = self.visual_U[i].to(self.device)
                            v_ = torch.mm(v.T, param.grad.data)
                            param.grad.data = torch.mm(v, v_) * self.lamda[int(name.split(".")[3])][i]

            elif isinstance(self._network, CLIP):

                for name, param in self._network.named_parameters():
                    for i in range(12):
                        if 'visual' in name and 'adapt' in name and 'down' in name and 'weight' in name:

                            v = self.visual_U[i].to(self.device)
                            v_ = torch.mm(param.grad.data, v)
                            param.grad.data = torch.mm(v_, v.T) * self.lamda[int(name.split(".")[3])][i]

                        elif 'visual' in name and 'adapt' in name and 'up' in name and 'weight' in name:

                            v = self.visual_U[i].to(self.device)
                            v_ = torch.mm(v.T, param.grad.data)
                            param.grad.data = torch.mm(v, v_) * self.lamda[int(name.split(".")[3])][i]


        return preds, acc, loss

    def inference(self, data, task_id = -1):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        if isinstance(self._network, CLIP):
            if task_id > -1:
                assert self.init_cls_num == self.inc_cls_num
                features_img, features_txt, logits_per_img, logits_per_txt = self._network(x, self.accm_text_tokens[task_id * self.inc_cls_num : (task_id + 1) * self.inc_cls_num])
            else:
                features_img, features_txt, logits_per_img, logits_per_txt = self._network(x, self.accm_text_tokens)
        elif isinstance(self._network, VIT):
            if task_id > -1:
                assert 0, 'Not Implemented'
            else:
                features = self._network(x)
                logits_per_img = []
                for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                    logits_per_img.append(prompts(features))
                logits_per_img = torch.cat(logits_per_img, dim=1)

        preds = logits_per_img.softmax(dim=-1).argmax(dim=1)

        if task_id > -1:
            assert self.init_cls_num == self.inc_cls_num
            preds += task_id * self.inc_cls_num

        acc = preds.eq(y).sum().item() / y.size(0)

        return preds, acc
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self._cur_task_id = task_idx
        if task_idx == 1:
            self._known_classes = self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        self.curr_class_names = train_loader.dataset.get_class_names()
        self.accm_class_names += self.curr_class_names

        self.curr_text_tokens = tokenize(
            [self.prompt_template.format(c) for c in self.curr_class_names]
        ).to(self.device)

        self.accm_text_tokens = tokenize(
            [self.prompt_template.format(c) for c in self.accm_class_names]
        ).to(self.device)

        if task_idx > 0:
            for data in train_loader:
                x = data['image'].to(self.device)
                self._network(x, self.curr_text_tokens, compute_lora_feat=True) # will replace last lora_feat

                for j in range(12): # Number of layers of both vision transformer and text transformer, hardcoded
                    activation_visual = self.visual_transformer_blocks[j].lora_feature
                    activation_visual = torch.bmm(activation_visual.permute(1, 2, 0),
                                                    activation_visual.permute(1, 0, 2)).sum(dim=0)
                    U_visual, _, _ = torch.linalg.svd(activation_visual, full_matrices=False)
                    U_visual = U_visual[:, 0:1]

                    for k in range(12):
                        v_visual = self.visual_U[k]
                        normalized_vector_visual = U_visual / torch.norm(U_visual)
                        similarities_visual = []

                        for column_visual in v_visual.t():
                            normalized_column_visual = column_visual / torch.norm(column_visual)
                            cos_sim_visual = torch.dot(normalized_vector_visual.squeeze(),
                                                        normalized_column_visual.squeeze())
                            similarities_visual.append(cos_sim_visual)

                        dot_products_visual = torch.mean(torch.topk(torch.stack(similarities_visual), int(len(similarities_visual) * 00.1))[0])
                        self.lamda[j][k] = torch.exp(-dot_products_visual) * self.lamda_scale

                break # first batch only

    @torch.no_grad()
    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        for data in train_loader:
            x = data['image'].to(self.device)
            self._network(x, self.curr_text_tokens, compute_lora_feat=True) # will replace last lora_feat

            for i in range(12):

                activation = self.visual_transformer_blocks[i].lora_feature
                
                activation = torch.bmm(activation.permute(1, 2, 0),
                                        activation.permute(1, 0, 2)).sum(dim=0)

                U, _, _ = torch.linalg.svd(activation, full_matrices=False)

                if task_idx == 0:
                    r = 0
                    self.visual_U.append(U[:,max(r,1):])
                else:
                    r = 1
                    Ui = torch.cat((self.visual_U[i], U[:, r:]), dim=1)
                    self.visual_U[i] = Ui

            break # first batch only

    def get_parameters(self, config):
        return self._network.parameters()