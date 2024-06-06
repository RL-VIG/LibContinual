'''
    We drew inspiration from the BIC original code and would like to express our gratitude to the outstanding contributors in the community whose code motivated us to refine the LibContinual framework.
    # Paper link https://arxiv.org/abs/1905.13260
    # Office link https://github.com/wuyuebupt/LargeScaleIncrementalLearning
    # Unoffice link https://github.com/sairin1202/BIC
'''
import os
path = os.getcwd()
os.chdir(path)

import torch
from torch import nn
import logging
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from core.model.backbone.resnet import BiasLayer
from core.data.dataset import BatchData, Exemplar
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from .finetune import Finetune
import torch.optim as optim
import copy


class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class, device):
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

class bic(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        # device setting
        self.device = kwargs['device']
        self.backbone = backbone
        bias_num = kwargs['task_num']
        self.bias_layers = []
        for i in range(bias_num):
            layer = BiasLayer().to(self.device)
            self.bias_layers.append(layer)
        self.model = Model(backbone, feat_dim, num_class, self.device)
        self.seen_cls = kwargs['init_cls_num']
        self.init_cls = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']
        self.task_num     = kwargs['task_num']
        self.T = 0
       
        optimizer_info = kwargs['optimizer']
        optimizer_name = optimizer_info['name']
        self.optimizer_kwargs = optimizer_info['kwargs']
        self.optimizer_cls = getattr(optim, optimizer_name)
        self.bias_optimizer = self.optimizer_cls(params=self.bias_layers[self.T].parameters(), **self.optimizer_kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.previous_model = None

     
        
    def bias_forward(self, input, bias_layers):
        outputs = []
        for i, layer in enumerate(bias_layers):
            if i == 0:
                input_slice = input[:, :self.init_cls]
            else :
                input_slice = input[:, (i-1) * self.inc_cls_num + self.init_cls:i* self.inc_cls_num + self.init_cls]
            output = layer(input_slice)
            outputs.append(output)
        combined_output = torch.cat(outputs, dim=1)  # Concatenating along the channel dimension

        return combined_output

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        pred = p[:,:self.seen_cls].argmax(dim=-1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0)
 
    
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        for i, layer in enumerate(self.bias_layers):
            layer.printParam(i)
        self.previous_model = deepcopy(self.model)
        self.previous_bias_layers = deepcopy(self.bias_layers)
        self.T += 1
        if self.T < self.task_num:
            self.bias_optimizer = self.optimizer_cls(params=self.bias_layers[self.T].parameters(), **self.optimizer_kwargs)
        self.seen_cls += self.inc_cls_num
    
    # The classic two-phase processing approach employed by BIC.
    def stage1(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        loss = self.criterion(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def stage1_distill(self, data):
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - self.inc_cls_num)/ self.seen_cls
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        with torch.no_grad():
            pre_p = self.previous_model(x)
            pre_p = self.bias_forward(pre_p,self.previous_bias_layers)
            pre_p = F.softmax(pre_p[:,:self.seen_cls-self.inc_cls_num]/T, dim=1)
        logp = F.log_softmax(p[:,:self.seen_cls-self.inc_cls_num]/T, dim=1)
        loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
        loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], y)
        loss = loss_soft_target + (1-alpha) * loss_hard_target
        distill_losses.append(loss_soft_target.item())
        ce_losses.append(loss_hard_target.item())        

        return pred, acc / x.size(0), loss


    def stage2(self, val_bias_data, bias_optimizer):
        losses = []
        x, y = val_bias_data['image'], val_bias_data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        
        loss = self.criterion(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        bias_optimizer.zero_grad()
        loss.backward()
        bias_optimizer.step()
        return pred, acc / x.size(0), loss

    def observe(self, data):
        if self.T > 0:
            return self.stage1_distill(data)
        else :
            return self.stage1(data)

    def bias_observe(self, data):
        return self.stage2(data, self.bias_optimizer)

    def get_parameters(self, config, stage2=False):
        if stage2:
            params = []
            for layer in self.bias_layers:
                params += layer.parameters()
            return params 
        else:
            return self.model.backbone.parameters()


    @staticmethod
    def split_data(dataloader, buffer, batch_size, task_idx):
        buffer_datasets = copy.deepcopy(dataloader.dataset)
        buffer_datasets.images=[]
        buffer_datasets.labels=[]    # empty

        train_datasets = copy.deepcopy(buffer_datasets) # empty
        val_datasets = copy.deepcopy(buffer_datasets)   # empty
        buffer_datasets.images.extend(buffer.images)   # buffer
        buffer_datasets.labels.extend(buffer.labels)
 
        from sklearn.model_selection import train_test_split
        images_train, images_val, labels_train, labels_val = train_test_split(buffer_datasets.images,
                                                                        buffer_datasets.labels,
                                                                        test_size=0.1,  
                                                                        random_state=42 
                                                                        )

        train_datasets.images.extend(images_train)   # 90% buffer
        train_datasets.labels.extend(labels_train)
        val_datasets.images.extend(images_val)       # 10% buffer
        val_datasets.labels.extend(labels_val)

        datasets = dataloader.dataset

        val_num = buffer.buffer_size/(task_idx*20)
        ratio = val_num/len(datasets.images)
        images_train, images_val, labels_train, labels_val = train_test_split(datasets.images,
                                                                        datasets.labels,
                                                                        test_size=ratio,  
                                                                        random_state=42 
                                                                        )

        train_datasets.images.extend(images_train)      # train = 90% current + 90% buffer
        train_datasets.labels.extend(labels_train)
        val_datasets.images.extend(images_val)          # val = 10% current + 10% buffer
        val_datasets.labels.extend(labels_val)          


        dataloader = DataLoader(
            train_datasets,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True
            )

        val_dataloader = DataLoader(
            val_datasets,
            shuffle=True,  
            batch_size=batch_size,
            drop_last=True
        )

        return dataloader, val_dataloader

