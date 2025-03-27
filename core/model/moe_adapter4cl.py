# -*- coding: utf-8 -*-
"""
@inproceedings{yu2024boosting,
  title={Boosting continual learning of vision-language models via mixture-of-experts adapters},
  author={Yu, Jiazuo and Zhuge, Yunzhi and Zhang, Lu and Hu, Ping and Wang, Dong and Lu, Huchuan and He, You},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23219--23230},
  year={2024}
}

Adapted from https://github.com/JiazuoYu/MoE-Adapters4CL
"""

import math
import torch
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from .backbone.clip import tokenize, CLIP
from .backbone.vit import ViTZoo

VIT = ViTZoo
CLIP = CLIP

class MOE_ADAPTER4CL(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.label_smoothing = kwargs['label_smoothing']

        self._known_classes = 0
        self._cur_task_id = -1

        self.accm_class_names = []   
        self.curr_class_names = []
        self.accm_text_tokens = None
        self.curr_text_tokens = None

        self.prompt_template = kwargs['prompt_template']

        self._network = backbone
        
        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)]
        )

        for name, param in self._network.named_parameters():
            if 'adaptmlp' not in name and 'router' not in name and 'noise' not in name:
                param.requires_grad = False

    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.
        '''

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        if isinstance(self._network, CLIP):
            features_img, features_txt, logits_per_img, logits_per_txt = self._network(x, self.curr_text_tokens)
        elif isinstance(self._network, VIT):
            features = self._network(x)
            logits_per_img = []
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits_per_img.append(prompts(features))
            logits_per_img = torch.cat(logits_per_img, dim=1)
        else:
            raise NotImplementedError
        
        loss = F.cross_entropy(logits_per_img, y, label_smoothing=self.label_smoothing)

        preds = logits_per_img.softmax(dim=-1).argmax(dim=1)
        acc = preds.eq(y).sum().item() / y.size(0)

        return preds, acc, loss

    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        if isinstance(self._network, CLIP):
            features_img, features_txt, logits_per_img, logits_per_txt = self._network(x, self.accm_text_tokens)
        elif isinstance(self._network, VIT):
            features = self._network(x)
            logits_per_img = []
            for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                logits_per_img.append(prompts(features))
            logits_per_img = torch.cat(logits_per_img, dim=1)
        else:
            raise NotImplementedError

        preds = logits_per_img.softmax(dim=-1).argmax(dim=1)
        acc = preds.eq(y).sum().item() / y.size(0)

        return preds, acc
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self._cur_task_id = task_idx
        if task_idx == 1:
            self._known_classes = self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        self.curr_class_names = train_loader.dataset.get_class_names()
        self.accm_class_names += self.curr_class_names

        self.curr_text_tokens = tokenize([self.prompt_template.format(c) for c in self.curr_class_names]).to(self.device)
        self.accm_text_tokens = tokenize([self.prompt_template.format(c) for c in self.accm_class_names]).to(self.device)

    def get_parameters(self, config):
        return self._network.parameters()