# -*- coding: utf-8 -*-
"""
@inproceedings{arXiv:2404.00228v3,
  title        = {InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning},
  author       = {Yan-Shuo Liang and Wu-Jun Li},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition, {CVPR} 2024, Seattle, Washington},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2024},
  url          = {https://arxiv.org/abs/2404.00228v3},
}
https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html

Adapted from https://github.com/liangyanshuo/InfLoRA?utm_source=catalyzex.com

Code Reference:
https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
"""

import math
import torch
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from .backbone.vit_inflora_opt import Attention_LoRA

class SiNet(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone

        assert kwargs["init_cls_num"] == kwargs["inc_cls_num"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs["init_cls_num"], bias=True)
            for _ in range(kwargs["task_num"])
        ])

    def update_fc(self):
        self._cur_task_id += 1

    def get_feature(self, x):
        features = self.backbone(x, task_id = self._cur_task_id)
        return features

    def forward(self, x, inference = False):
        logits = []
        features = self.backbone(x, task_id = self._cur_task_id)
        if inference:
            for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                logits.append(prompts(features))
        else:
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits.append(prompts(features))

        return torch.cat(logits, dim=1)

    def update_input_matrix(self, x):
        self.backbone(x, task_id = self._cur_task_id, get_input_matrix = True)

class InfLoRA_OPT(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]

        self._known_classes = 0
        self.feature_list = []
        self.project_type = []

        self._network = SiNet(backbone, **kwargs)

        self.attention_modules = []
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                self.attention_modules.append(module)

        self._network.to(self.device)

    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.
        '''

        # Masked Learned Classes
        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc, loss
    
    def inference(self, data):
        '''
        It is called in the inference phase to input a batch of test samples and return the classification result and accuracy. 
        Calling the interface function of _network returns the value batchsize*_total_classes.
        '''

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logits = self._network(x, inference = True)
        preds = logits.max(1)[1]

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        It is called before the training of each task to update the parameters, select the branch for training, and update the lora_A matrix of the corresponding branch
        '''

        if task_idx == 1:
            self._known_classes = self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num
        self._network.update_fc()

        for module in self.attention_modules:
            module.init_param()

        # Freeze the model and only train the linear layer, and lora_b layer
        unfrezeed_params = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            
            # impl 2
            if "classifier_pool." + str(task_idx) in name or "lora_B" in name:
                param.requires_grad_(True)
                unfrezeed_params.append(name)
            

            ''' impl 1
            if "classifier_pool." + str(task_idx) in name or "lora_B_k." + str(task_idx) in name or "lora_B_v." + str(task_idx) in name:
                param.requires_grad_(True)
                unfrezeed_params.append(name)
            '''

        print(f"Current task : {task_idx}, Parameters to be updated: {len(unfrezeed_params)}")
        print(",".join(unfrezeed_params))

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):
                x = batch['image'].to(self.device)
                self._network.update_input_matrix(x)

        if task_idx == 0:
            for module in self.attention_modules:
                U, _, _ = torch.linalg.svd(module.cur_matrix)
                module.lora_A_k.weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                module.reset_input_matrix()
        else:
            for i, module in enumerate(self.attention_modules):
                assert self.project_type[i] == 'remove' or self.project_type[i] == 'retain'

                cur_matrix = module.cur_matrix
                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T)

                if self.project_type[i] == 'remove':
                    cur_matrix = cur_matrix - feature_mat @ cur_matrix
                else:
                    cur_matrix = feature_mat @ cur_matrix

                U, _, _ = torch.linalg.svd(cur_matrix, full_matrices = False)
                module.lora_A_k.weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                module.reset_input_matrix()
    

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        for module in self.attention_modules:
            module.merge_weight()

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):
                x = batch['image'].to(self.device)
                self._network.update_input_matrix(x)

        self.update_DualGPM(task_idx)

    def update_DualGPM(self, task_idx):
        '''
        Update feature lists and the corresponding type
        '''

        threshold = (self.lame - self.lamb) * task_idx/self.task_num + self.lamb
        print(f'Threshold: {threshold}')

        if task_idx == 0:
            for module in self.attention_modules:
                activation = module.cur_matrix
                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = max(np.sum(np.cumsum(sval_ratio) < threshold), 1)
                self.feature_list.append(U[: ,:r])
                if r < (activation.shape[0]/2):
                    self.project_type.append('remove')
                else:
                    self.project_type.append('retain')

                module.reset_input_matrix()
        else:
            for i, module in enumerate(self.attention_modules):
                activation = module.cur_matrix
                _, S, _ = np.linalg.svd(activation, full_matrices = False)
                sval_total = (S**2).sum()

                if self.project_type[i] == 'remove':
                    act_hat = activation - torch.Tensor(self.feature_list[i] @ self.feature_list[i].T) @ activation
                    U, S, _ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total

                    if accumulated_sval >= threshold:
                        print (f'Skip Updating DualGPM for layer: {i+1}')
                    else:
                        r = np.sum(np.cumsum(sval_ratio) + accumulated_sval < threshold) + 1
                        Ui = np.hstack((self.feature_list[i], U[:, :r]))  
                        self.feature_list[i] = Ui[:, :min(Ui.shape[0], Ui.shape[1])]
            
                else:
                    act_hat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T) @ activation
                    U, S, _ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total        
                    accumulated_sval = sval_hat/sval_total       

                    if accumulated_sval < 1 - threshold:
                        print (f'Skip Updating Space for layer: {i+1}')
                    else:
                        r = np.sum(accumulated_sval - np.cumsum(sval_ratio) >= 1 - threshold) + 1
                        act_feature = self.feature_list[i] - U[:,0:r] @ U[:,0:r].T @ self.feature_list[i]
                        U, _, _ = np.linalg.svd(act_feature)
                        self.feature_list[i] = U[:, :self.feature_list[i].shape[1]-r]

                module.reset_input_matrix()

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

    def get_parameters(self, config):
        return self._network.parameters()