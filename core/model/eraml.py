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

    def return_hidden(self, data):
        return self.backbone(data)

    def forward(self, data):
        return self.classifier(self.backbone(data))
    
class ERAML(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.model = Model(backbone, kwargs['num_classes'])
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.use_augs = kwargs['use_augs']
        self.supcon_temperature = kwargs['supcon_temperature']
        self.use_minimal_selection = kwargs['use_minimal_selection']
        self.task_free = kwargs['task_free']
        self.device = device

        self.sample_kwargs = {
            'amt': 10,
            'exclude_task': None
        }

        self.model.to(self.device)

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def sup_con_loss(self, anchor_feature, features, anch_labels=None, labels=None,
                    mask=None, temperature=0.1, base_temperature=0.07):

        batch_size, anchor_count, _ = features.shape

        labels = labels.contiguous().view(-1, 1)           
        anch_labels = anch_labels.contiguous().view(-1, 1) 
        mask = torch.eq(anch_labels, labels.T).float().to(self.device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # hid_all

        # compute logits
        anchor_dot_contrast = torch.div(anchor_feature @ contrast_feature.T, temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) 
        logits = anchor_dot_contrast - logits_max.detach() 

        # tile mask
        mask = mask.repeat(anchor_count, anchor_count)

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

        x, y = inc_data['x'], inc_data['y']

        logits = self.model(x)
        pred = logits.max(1)[1]

        # If task_based, see if task id >= 1
        # If task_free, see if buffer has something
        if inc_data['t'] > 0 or (self.task_free and len(self.buffer) > 0):
            pos_x, neg_x, pos_y, neg_y, invalid_idx, _ = self.sample(
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

            loss = 0.
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
                    temperature=self.supcon_temperature
                ) 
                
        else:
            # do regular training at the start
            loss = F.cross_entropy(logits, y)

        correct_count = (pred == y).sum().item()

        return pred, correct_count, loss

    def observe(self, data):

        inc_correct_counts, inc_total_counts, re_correct_counts, re_total_counts  = 0, 0, 0, 0

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        self.inc_data = {'x': x, 'y': y, 't': self.cur_task_idx}

        pred, correct_count, loss = self.process_inc(self.inc_data)
        total_count = y.shape[0]

        if len(self.buffer) > 0 and (self.task_free or self.cur_task_idx > 0):
            re_data = self.buffer.sample(**self.sample_kwargs)

            re_logits = self.model(re_data['x'])
            loss += F.cross_entropy(re_logits, re_data['y'])
            re_pred = re_logits.max(1)[1]
            correct_count += (re_pred == re_data['y']).sum().item()
            total_count += re_data['y'].shape[0]
        
        acc = correct_count / total_count

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
        if self.use_minimal_selection:
            self.sample = self.buffer.sample_minimal_pos_neg
        else:
            self.sample = self.buffer.sample_pos_neg

        self.cur_task_idx = task_idx

    def add_reservoir(self):
        self.buffer.add(self.inc_data)

    def get_parameters(self, config):
        return self.model.parameters()
