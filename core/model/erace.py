# -*- coding: utf-8 -*-
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
import torch.nn.functional as F

from torch import nn

class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def return_hidden(self, data):
        return self.backbone(data)['features']
    
    def forward(self, data):
        return self.classifier(self.backbone(data)['features'])
    
class ERACE(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.model = Model(backbone, kwargs['num_classes'])
        self.device = device
        self.loss = F.cross_entropy

        # note that this is not used for task-free methods
        self.task = 0
        self.cur_task_idx = -1
        self.inc_data = {}

        # hardcoded for now
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)

        self.task_free = kwargs['task_free']

        self.sample_kwargs = {
            'amt':          10,
            'exclude_task': None
        }

        self.buffer_exists = False

        self.model.to(self.device)

    def process_inc(self, inc_data):
        """ get loss from incoming data """

        present = inc_data['y'].unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        # process data
        logits = self.model(inc_data['x'])
        mask = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        mask[:, self.seen_so_far.max():] = 1

        if inc_data['t'] > 0 or self.task_free:
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss = self.loss(logits, inc_data['y'].long())
        pred = logits.max(1)[1]
        correct_count = pred.eq(inc_data['y']).sum().item()

        return pred, correct_count, loss

    def process_re(self, re_data):
        logits = self.model(re_data['x'])
        loss = self.loss(logits, re_data['y'].long())
        pred   = logits.max(1)[1]
        correct_count = pred.eq(re_data['y']).sum().item()

        return correct_count, loss

    def observe(self, data):

        inc_correct_counts, inc_total_counts, re_correct_counts, re_total_counts = 0, 0, 0, 0

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        self.inc_data = {'x': x, 'y': y, 't': self.cur_task_idx}

        # keep track of current task for task-based methods
        self.task = self.cur_task_idx

        
        pred, inc_correct_counts, inc_loss = self.process_inc(self.inc_data)
        inc_total_counts = self.inc_data['y'].shape[0]

        
        re_loss = 0
        if len(self.buffer) > 0 and (self.task_free or self.task > 0):
            re_data = self.buffer.sample_random(**self.sample_kwargs)
            re_correct_counts, re_loss = self.process_re(re_data)
            re_total_counts = re_data['y'].shape[0]

        total_loss = inc_loss + re_loss

        acc = (inc_correct_counts + re_correct_counts) / (inc_total_counts + re_total_counts)

        self.buffer.add_reservoir(self.inc_data)

        # only return output of incoming data, not including output of rehearsal data
        return pred, acc, total_loss

    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self.model(x)
        pred   = logits.max(1)[1]
        correct_count = pred.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return pred, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if (self.buffer_exists == False):
            self.buffer = buffer
            self.buffer.set_device(self.device)
            self.buffer_exists = True

        self.cur_task_idx = task_idx

    def get_parameters(self, config):
        return self.model.parameters()



