# -*- coding: utf-8 -*-
"""
@inproceedings{wu2019large,
  title={Large Scale Incremental Learning},
  author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={374--382},
  year={2019}
}
https://arxiv.org/abs/1905.13260

Adapted from https://github.com/wuyuebupt/LargeScaleIncrementalLearning and https://github.com/sairin1202/BIC.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from copy import deepcopy
from torch.utils.data import DataLoader
from core.model.backbone.resnet import BiasLayer
from .finetune import Finetune
from sklearn.model_selection import train_test_split

class Model(nn.Module):

    def __init__(self, backbone, feat_dim, num_class, device):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
    
    def forward(self, x):
        return self.classifier(self.backbone(x)['features'])
    
class bic(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        #super().__init__(backbone, feat_dim, num_class, **kwargs)
        super().__init__()

        self.backbone = backbone
        self.device = kwargs['device']
        self.task_num = kwargs['task_num']
        self.bias_layers = [BiasLayer().to(self.device) for _ in range(self.task_num)]

        self.model = Model(backbone, feat_dim, num_class, self.device)
        self.init_cls = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']
        self.seen_cls = self.init_cls

        self.T = 0

        #optimizer_info = kwargs['optimizer']
        #optimizer_name = optimizer_info['name']
        #self.optimizer_kwargs = optimizer_info['kwargs']
        #self.optimizer_cls = getattr(optim, optimizer_name)
        #self.bias_optimizer = self.optimizer_cls(params=self.bias_layers[self.T].parameters(), **self.optimizer_kwargs)
        self.bias_optimizer = optim.Adam(self.bias_layers[self.T].parameters(), lr=0.001)
        self.previous_model = None
 
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        for layer in self.bias_layers:
            layer.eval()

    def bias_forward(self, input, bias_layers):
        outputs = []
        for i, layer in enumerate(bias_layers):
            if i == 0:
                input_slice = input[:, :self.init_cls]
            else:
                input_slice = input[:, (i-1) * self.inc_cls_num + self.init_cls : i * self.inc_cls_num + self.init_cls]
            outputs.append(layer(input_slice))

        return torch.cat(outputs, dim=1)

    def inference(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        pred = p[:,:self.seen_cls].argmax(dim=-1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0)
     
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        for i, layer in enumerate(self.bias_layers):
            layer.train()
            layer.printParam(i)
        self.previous_model = deepcopy(self.model)
        self.previous_bias_layers = deepcopy(self.bias_layers)
        self.T += 1
        if self.T < self.task_num:
            self.bias_optimizer = optim.Adam(self.bias_layers[self.T].parameters(), lr=0.001)
        self.seen_cls += self.inc_cls_num
    
    # The classic two-phase processing approach employed by BIC.
    def stage1(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        loss = nn.CrossEntropyLoss()(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def stage1_distill(self, data):
        '''
        Code Reference:
        https://github.com/sairin1202/BIC/blob/master/trainer.py
        '''

        T = 2
        alpha = (self.seen_cls - self.inc_cls_num)/ self.seen_cls
        x, y = data['image'].to(self.device), data['label'].to(self.device)
        p = self.model(x)
        p = self.bias_forward(p,self.bias_layers)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        with torch.no_grad():
            pre_p = self.previous_model(x)
            #pre_p = self.bias_forward(pre_p,self.previous_bias_layers)
            pre_p = self.bias_forward(pre_p, self.bias_layers)
            pre_p = F.softmax(pre_p[:,:self.seen_cls-self.inc_cls_num]/T, dim=1)
        logp = F.log_softmax(p[:,:self.seen_cls-self.inc_cls_num]/T, dim=1)
        loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
        loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], y)
        loss = loss_soft_target + (1-alpha) * loss_hard_target
    
        return pred, acc / x.size(0), loss

    def stage2(self, val_bias_data):

        x, y = val_bias_data['image'].to(self.device), val_bias_data['label'].to(self.device)
        p = self.model(x)
        p = self.bias_forward(p, self.bias_layers)
        
        loss = nn.CrossEntropyLoss()(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        self.bias_optimizer.zero_grad()
        loss.backward()
        self.bias_optimizer.step()

        return pred, acc / x.size(0), loss

    def observe(self, data):
        if self.T > 0:
            return self.stage1_distill(data)
        else :
            return self.stage1(data)

    def bias_observe(self, data):
        return self.stage2(data)

    def get_parameters(self, config, stage2=False):
        if stage2:
            params = []
            for layer in self.bias_layers:
                params += layer.parameters()
            return params 
        else:
            return self.model.backbone.parameters()

    @staticmethod
    def split_data(dataloader, buffer, task_idx, config):

        # -----

        buffer_datasets = deepcopy(dataloader.dataset)

        buffer_datasets.images=[]
        buffer_datasets.labels=[]    # empty


        train_datasets = deepcopy(buffer_datasets) # empty
        val_datasets = deepcopy(buffer_datasets)   # empty
        buffer_datasets.images.extend(buffer.images)   # buffer
        buffer_datasets.labels.extend(buffer.labels)
 
        
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
            num_workers=num_workers,
            drop_last=True
            )

        val_dataloader = DataLoader(
            val_datasets,
            shuffle=True,  
            batch_size=100,
            num_workers=num_workers,
            drop_last=True
        )

        return dataloader, val_dataloader

    @staticmethod
    def get_train_loader(dataloader, buffer, task_idx, config):

        images_train, _, labels_train, _ = train_test_split(buffer.images,
                                                            buffer.labels,
                                                            test_size=0.1,  # 9 : 1
                                                            random_state=42)

        train_dataset = deepcopy(dataloader.dataset)

        print('train', f'{len(train_dataset.images)} + {len(images_train)}')

        train_dataset.images.extend(images_train)
        train_dataset.labels.extend(labels_train)

        return DataLoader(
            train_dataset,
            shuffle=True,  
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=False)

    @staticmethod
    def get_val_bias_loader(dataloader, buffer, task_idx, config):

        _, images_val, _, labels_val = train_test_split(buffer.images,
                                                        buffer.labels,
                                                        test_size=0.1,  # 9 : 1
                                                        random_state=42)

        val_bias_dataset = deepcopy(dataloader.dataset)

        val_bias_dataset.images = images_val
        val_bias_dataset.labels = labels_val

        print('buffer', len(buffer.images))
        print('val', len(images_val))

        return DataLoader(
            val_bias_dataset,
            shuffle=True,  
            batch_size=100,
            num_workers=config['num_workers'],
            drop_last=False)