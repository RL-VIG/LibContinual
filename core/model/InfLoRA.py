# -*- coding: utf-8 -*-
"""
@inproceedings{arXiv:2404.00228v3,
  title        = {InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning},
  author       = {Yan-Shuo Liang and
                  Wu-Jun Li},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition, {CVPR} 2024, Seattle, Washington},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2024},
  url          = {https://arxiv.org/abs/2404.00228v3},
}
https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html

Adapted from https://github.com/liangyanshuo/InfLoRA?utm_source=catalyzex.com
"""


import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from .backbone.vit_inflora import Attention_LoRA
from copy import deepcopy
import math
from  .finetune import Finetune


class InfLoRA(Finetune):

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)

        self._network = backbone

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        # 100 categories in total, parameter passed assignment
        self.num_class = num_class
        # number of known and number of classes
        self._total_classes =0
        # Number of categories known before this task, initially 0, updated in beforetask
        self._known_classes =0

        # The current task number, initially -1. +1 for each new task
        self._cur_task = -1
        # number of tasks incremented each time
        self.inc_cls_num = kwargs["inc_cls_num"]
        
        self.device = kwargs["device"]

        # These parameters are used in update DualGPM
        self.feature_list = []
        self.project_type = []
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]
        self.total_sessions = kwargs["total_sessions"]
        
    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.

        Code Reference:
        https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
        '''
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # Offset the target because the forward function in _network only predicts 0-9
        y = y-self._known_classes  
        logits = self._network(x)['logits']
        loss = F.cross_entropy(logits, y)
        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(y.expand_as(preds)).cpu().sum()
        total = len(y)
        acc = correct/total
        acc = acc.item()
        return preds, acc, loss
    
    def inference(self, data):
        '''
        It is called in the inference phase to input a batch of test samples and return the classification result and accuracy. 
        Calling the interface function of _network returns the value batchsize*_total_classes.

        Code Reference:
        https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
        '''
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self._network.interface(x)
        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(y.expand_as(preds)).cpu().sum()
        total = len(y)
        acc = correct/total
        acc = acc.item()
        return preds, acc
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        It is called before the training of each task to update the parameters, select the branch for training, and update the lora_A matrix of the corresponding branch

        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
        '''

        # Update some variables
        self._known_classes = self._total_classes       
        self._cur_task += 1
        self._total_classes = self._known_classes + self.inc_cls_num
        self._network.update_fc(self._total_classes)

        self._network.to(self.device)
        
        # Freeze the model and only release the linear layer, and the lora_b layer corresponding to the task number to train
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        # Check the layer to be trained
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        with torch.no_grad():
            # We run the trained data through the model in order to obtain the cur_matrix. This parameter is related to update_DualGPM
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch["image"]
                targets = batch["label"]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs=F.interpolate(inputs, size=224, mode='bilinear', align_corners=False)
                self._network(inputs, get_cur_feat=True)
                
            if self._cur_task == 0:
                # Updating according to cur matrix requires A manually designed lora A
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                # Updating according to cur matrix requires A manually designed lora A
                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                        else:
                            assert self.project_type[kk] == 'retain'
                            cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1
    
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task starts training, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''
        with torch.no_grad():
            # Get cur_matrix
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch["image"]
                targets = batch["label"]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs=F.interpolate(inputs, size=224, mode='bilinear', align_corners=False)
                self._network(inputs, get_cur_feat=True)
            # Preliminary operations on the mapping matrix
            mat_list = []
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            self.update_DualGPM(mat_list)
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf=torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                self.feature_mat.append(Uf)
        
        return

    def update_DualGPM (self, mat_list):
        '''
        Code Reference:
        https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
        '''
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)
        
    def _set_random(self,args):
        '''
        Set random values on various devices to ensure repeatable results
        '''
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False