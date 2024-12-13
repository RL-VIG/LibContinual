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
import torch.nn as nn

#from torch import nn
from collections import OrderedDict
from collections.abc import Iterable

class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def return_hidden(self, data):
        return self.backbone(data)['features']
    
    def forward(self, data):
        return self.classifier(self.backbone(data)['features'])
    
class ERAML(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.model = Model(backbone, kwargs['num_classes'])
        self.device = device
        self.loss = F.cross_entropy

        # note that this is not used for task-free methods
        self.task = 0
        self.cur_task_idx = -1
        self.inc_data = {}

        self.task_free = kwargs['task_free']

        self.sample_kwargs = {
            'amt':          10,
            'exclude_task': None
        }

        self.buffer_exists = False

        self.model.to(self.device) # This is not working

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def sup_con_loss(self, anchor_feature, features, anch_labels=None, labels=None,
                    mask=None, temperature=0.1, base_temperature=0.07):

        device = features.device

        if features.ndim < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None:  
            labels = labels.contiguous().view(-1, 1)           
            anch_labels = anch_labels.contiguous().view(-1, 1) 
            if labels.shape[0] != batch_size:
                print(f"len of labels: {len(labels)}")
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(anch_labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # hid_all

        # anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) 
        logits = anchor_dot_contrast - logits_max.detach() 

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def process_inc(self, inc_data):
        """ get loss from incoming data """

        """
        inc_data = {'x':Tensor[samples_per_task, feature_dimension],
                    'y':Tensor[samples_per_task]},
                    't':task_idx
        x：样本 tensor
        y: 标签 tensor
        """

        n_fwd = 0
        correct_count = 0

        x, y = inc_data['x'], inc_data['y']

        logits = self.model(x)
        pred   = logits.max(1)[1]

        # If task_based, see if task id >= 1
        # If task_free, see if buffer has something
        if inc_data['t'] > 0 or (self.task_free and len(self.buffer) > 0):
            pos_x, neg_x, pos_y, neg_y, invalid_idx, n_fwd = \
                    self.buffer.sample_pos_neg(
                        inc_data,
                        task_free = self.task_free,
                        same_task_neg = True # If true, neg sample can only choose from inc_data, instead of inc_data + buffer
                    )
            
         
            hidden  = self.model.return_hidden(inc_data['x'])   
            hidden_norm = self.normalize(hidden[~invalid_idx])  

            all_xs = torch.cat((pos_x, neg_x))
            all_hid = self.normalize(self.model.return_hidden(all_xs))
            all_hid = all_hid.reshape(2, pos_x.size(0), -1)
            pos_hid, neg_hid = all_hid[:, ~invalid_idx]        

            if (~invalid_idx).any():
                inc_y = y[~invalid_idx]             
                pos_y = pos_y[~invalid_idx]                    
                neg_y = neg_y[~invalid_idx]                   
                hid_all = torch.cat((pos_hid, neg_hid), dim=0)
                y_all   = torch.cat((pos_y, neg_y), dim=0)
                
                loss = self.sup_con_loss(
                        labels=y_all,                             
                        features=hid_all.unsqueeze(1),             
                        anch_labels=inc_y.repeat(2),              
                        anchor_feature=hidden_norm.repeat(2, 1),   
                        temperature=0.2, #hardcoded for now
                ) 
            else:
                loss = 0.

        else:
            # do regular training at the start
            loss = self.loss(logits, y.long())


        correct_count = pred.eq(y).sum().item()

        return pred, correct_count, loss

    def process_re(self, re_data):
        logits = self.model(re_data['x'])

        loss = self.loss(logits, re_data['y'].long())
        pred   = logits.max(1)[1]
        correct_count = pred.eq(re_data['y']).sum().item()

        return correct_count, loss

    def observe(self, data):

        inc_correct_counts, inc_total_counts, re_correct_counts, re_total_counts  = 0, 0, 0, 0

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        
        self.inc_data = {'x': x, 'y': y, 't': self.cur_task_idx}

        # keep track of current task for task-based methods
        self.task = self.cur_task_idx

        # 处理新数据
        pred, inc_correct_counts, inc_loss = self.process_inc(self.inc_data)
        inc_total_counts = self.inc_data['y'].shape[0]

        # 处理新+旧数据
        re_loss  = 0
        if len(self.buffer) > 0 and (self.task_free or self.task > 0):
            re_data = self.buffer.sample_random(**self.sample_kwargs)
            re_correct_counts, re_loss = self.process_re(re_data)
            re_total_counts = re_data['y'].shape[0]
        
        total_loss = inc_loss + re_loss

        acc = (inc_correct_counts + re_correct_counts) / (inc_total_counts + re_total_counts)

        self.buffer.add_reservoir(self.inc_data)

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


