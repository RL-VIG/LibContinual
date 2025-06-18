"""
@misc{caccia2022new,
    title={New Insights on Reducing Abrupt Representation Change in Online Continual Learning}, 
    author={Lucas Caccia and Rahaf Aljundi and Nader Asadi and Tinne Tuytelaars and Joelle Pineau and Eugene Belilovsky},
    year={2022},
    eprint={2104.05025},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

Adapted from https://github.com/pclucas14/AML
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class distLinear(nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super().__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        cos_dist = torch.mm(x_normalized,self.L.weight.div(L_norm + 0.00001).transpose(0,1))

        scores = self.scale_factor * (cos_dist)

        return scores

class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = distLinear(160, num_classes)

    def forward(self, data):
        return self.classifier(self.backbone(data))
    
class ERACE(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.model = Model(backbone, kwargs['num_classes'])
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.use_augs = kwargs['use_augs']
        self.device = device
        self.seen_so_far = 0

        self.task_free = kwargs['task_free']
        assert self.task_free, 'ER-ACE must be task free'

        self.sample_kwargs = {
            'amt':          10,
            'exclude_task': None
        }

        self.model.to(self.device)

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        self.inc_data = {'x': x, 'y': y, 't': self.cur_task_idx}

        logits = self.model(x)

        mask = torch.zeros_like(logits)
        mask[:, self.seen_so_far:] = 1

        if self.cur_task_idx > 0 or self.task_free:
            logits = logits.masked_fill(mask == 0, -1e9)

        loss = F.cross_entropy(logits, y)
        pred = logits.max(1)[1]
        correct_count = (pred == y).sum().item()
        total_count = y.shape[0]

        if len(self.buffer) > 0 and (self.task_free or self.cur_task_idx > 0):
            re_data = self.buffer.sample_random(**self.sample_kwargs)

            re_logits = self.model(re_data['x'])
            loss += F.cross_entropy(re_logits, re_data['y'])
            re_pred = re_logits.max(1)[1]
            correct_count += (re_pred == re_data['y']).sum().item()
            total_count += re_data['y'].shape[0]

        acc = correct_count / total_count

        # only return output of incoming data, not including output of rehearsal data
        return pred, acc, loss

    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self.model(x)
        pred   = logits.max(1)[1]
        correct_count = pred.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return pred, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if not self.use_augs:
            train_loader.dataset.trfms = test_loaders[0].dataset.trfms

        self.buffer = buffer
        self.buffer.device = self.device

        self.cur_task_idx = task_idx

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self.seen_so_far = self.init_cls_num + self.inc_cls_num * task_idx

    def add_reservoir(self):
        self.buffer.add_reservoir(self.inc_data)

    def get_parameters(self, config):
        return self.model.parameters()
