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



class EWC(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        # self.classifier = CosineLinear(feat_dim, kwargs['init_cls_num'])
        self.fisher = None
        self.lamda = self.kwargs['lamda']
        
        self.test_fc_a = nn.Linear(10, 10, bias=False)
        self.test_fc_b = nn.Linear(5, 5, bias=False)
        
    def get_parameters(self,  config):

        # # case1
        # train_parameters = []
        # train_parameters.append({"params": self.test_fc_a.parameters(), "lr": 0.1, "weigth_decay": 0.00005})
        # train_parameters.append({"params": self.test_fc_b.parameters(), "lr": 0.01})
        
        # case2
        train_parameters = []
        train_parameters.append({"params": self.backbone.parameters()})
        return train_parameters
        


    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        in_features = self.backbone.fc.in_features
        out_features = self.backbone.fc.out_features
        
        new_fc = nn.Linear(in_features, self.kwargs['init_cls_num'] + task_idx * self.kwargs['inc_cls_num'])
        new_fc.weight.data[:out_features] = self.backbone.fc.weight.data
        self.backbone.fc = new_fc
        self.backbone.to(self.device)


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.shape)
        # logit = self.classifier(self.backbone(x)['features'])    
        logit = self.backbone(x)
        if self.task_idx == 0:
            loss = F.cross_entropy(logit, y)
        else:
            old_classes = self.backbone.fc.out_features - self.kwargs['inc_cls_num']
            loss = F.cross_entropy(logit[:, old_classes:], y - old_classes)
            loss += self.lamda * self.compute_ewc()
            
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss


    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx == 0:
            self.fisher = self.getFisher(train_loader)
        else:
            alpha = 1 - self.kwargs['inc_cls_num']/self.backbone.fc.out_features
            new_finsher = self.getFisher(train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                    alpha * self.fisher[n]
                    + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
            
        self.mean = {
            n: p.clone().detach()
            for n, p in self.backbone.named_parameters()
            if p.requires_grad
        }
        
        for n in self.mean.keys():
            self.mean[n].to(self.device)
        
    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logit = self.backbone(x)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)
    

    def getFisher(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self.device)
            for n, p in self.backbone.named_parameters()
            if p.requires_grad
        }
        self.backbone.train()
        optimizer = optim.SGD(self.backbone.parameters(), lr=0.1)
        for batch_idx, data in enumerate(train_loader):
            x, y = data['image'], data['label']
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits = self.backbone(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(0.0001))
        return fisher

    def compute_ewc(self):
        loss = 0
        for n, p in self.backbone.named_parameters():
            if n in self.fisher.keys():
                # print(p.device)
                # print(self.mean[n].device)
                loss += (
                    torch.sum(
                        (self.fisher[n])
                        * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                    )
                    / 2
                )
        return loss