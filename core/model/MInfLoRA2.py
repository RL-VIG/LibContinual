"""
Code Reference:
https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
from math import pi
from torchvision import transforms

from .backbone.transformer import MultiHeadAttention_MultiMaskedLoRA

Epsilon = 0.5

class TopK:

    '''
    A class to maintain a collection of the top K items based on a specified attribute.

    This class allows for the dynamic addition of items, each represented as a dictionary, 
    where each dictionary must have a key 'proj_norm' that represents the value used 
    to determine the ranking. The class keeps track of the top K items with the highest 
    'proj_norm' values.
    '''

    def __init__(self, k):
        self.k = k
        self.top_k_list = []

    def add(self, dict):
        if len(self.top_k_list) < self.k:
            self.top_k_list.append(dict)
        elif dict['proj_norm'] > min(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm']:
            self.top_k_list.remove(min(self.top_k_list, key=lambda x: x['proj_norm']))
            self.top_k_list.append(dict)
        elif dict['proj_norm'] == min(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm'] and \
            dict['proj_norm'] == max(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm']:
            self.top_k_list.remove(min(self.top_k_list, key=lambda x: x['task_id']))
            self.top_k_list.append(dict)

    def get_top_k(self):
        return self.top_k_list

class SiNet(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]

        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)])

        for name, module in self.backbone.named_modules():
            if 'transformer' in name and 'blocks' not in name:
                self.transformer_module = module

    def update_fc(self):
        self._cur_task_id += 1

    def fc_only(self, x, expert_id):
        logits = []
        for prompts in self.classifier_pool[:expert_id + 1]:
            logits.append(prompts(x))
        return torch.cat(logits, dim=1)
    
    def fc_only2(self, x):
        logits = []
        for prompts in self.classifier_pool[:self._cur_task_id + 1]:
            logits.append(prompts(x))
        return torch.cat(logits, dim=1)

    def get_feature(self, x, expert_id):
        features = self.backbone(x, expert_id = expert_id)
        return features

    def forward(self, x, expert_id, inference = False):
        logits = []
        features = self.backbone(x, expert_id = expert_id)

        if inference:

            probs = self.transformer_module.probs
            probs = torch.Tensor(probs[-1]).to(x.device) # consider only last layer

            # Bayesian
            for i, prompts in enumerate(self.classifier_pool[:self._cur_task_id + 1]):
                logits.append(prompts(features))

            logits = torch.cat(logits, dim=1)

            return logits

        else:
            logits.append(self.classifier_pool[self._cur_task_id](features))
            return torch.cat(logits, dim=1)

    def update_input_matrix(self, x):
        self.backbone(x, expert_id = 0, get_input_matrix = True)

class MInfLoRA2(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]
        self.eval_mat = kwargs['eval_mat']

        self._known_classes = 0
        self.feature_list = []
        self.project_type = []

        self._network = SiNet(backbone, **kwargs)

        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_MultiMaskedLoRA)]

        # TRGP Implementation
        self.feature_list_each_tasks = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]
        self.final_decision = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]
        self.before_mat = [[0 for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]

        self.experts_distributions = []

        # Class Alignment Implementation
        self._use_class_alignment = kwargs['use_ca']
        self._class_means = None
        self._class_covs = None
        self._dataset = kwargs['dataset']
        if self._dataset == 'cifar':
            self.logit_norm = None
        else:
            self.logit_norm = 0.1   
    
        self.lll = []

        self._network.to(self.device)
        
    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.
        '''

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(x, expert_id = self._network._cur_task_id) # hardcoded for task_id
        loss = F.cross_entropy(logits, y)

        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc, loss
    
    def inference(self, data, **kwargs):

        task_id = kwargs['task_id'] if 'task_id' in kwargs else None
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self._network(x, expert_id = 0, inference = True)
        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc

    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num
        self._network.update_fc()

        for module in self.attention_modules:
            module.init_param()

        self._update_input_matrix(train_loader, test_loaders[0].dataset.trfms)

        for i, module in enumerate(self.attention_modules):
            
            topk = TopK(1)

            mat = module.cur_matrix.cpu().numpy()
            mat_norm = np.linalg.norm(mat)

            for task_id in range(task_idx):
            
                proj_norm = np.linalg.norm(self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T @ mat)
                
                if proj_norm > Epsilon * mat_norm:
                    topk.add({'proj_norm':proj_norm, 'task_id': task_id})

            self.final_decision[task_idx][i] = [dic['task_id'] for dic in topk.get_top_k()]

            module.enable_scale(task_id = task_idx, space = [torch.tensor(self.feature_list_each_tasks[task_id][i]).to(self.device) for task_id in self.final_decision[task_idx][i]])
            print(f'Layer {i} of {task_idx} consider {self.final_decision[task_idx][i]} as trust region')
            
        if task_idx == 0:
            for i, module in enumerate(self.attention_modules):
                U, _, _ = torch.linalg.svd(module.cur_matrix)
                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
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

                U, _, _ = np.linalg.svd(cur_matrix.cpu().numpy(), full_matrices = False)
                U = torch.tensor(U).to(self.device)

                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.reset_input_matrix()

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if f"classifier_pool.{task_idx}" in name or f"lora_B" in name or f"scale_param.{task_idx}" in name:
                param.requires_grad_(True)
        unfrezeed_params = [name for name, param in self._network.named_parameters() if param.requires_grad]

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        [module.merge_weight() for module in self.attention_modules]

        self._update_feature(task_idx, train_loader, test_loaders)

        self._update_input_matrix(train_loader, test_loaders[0].dataset.trfms)

        threshold = self.lamb

        for i, module in enumerate(self.attention_modules):

            activation = module.cur_matrix
            U, S, _ = np.linalg.svd(activation, full_matrices=False)
            sval_ratio = (S**2)/(S**2).sum()

            r = max(np.sum(np.cumsum(sval_ratio) < threshold), 1)

            # DEBUG, REMOVE
            tnsr = torch.Tensor(U[:, :r])
            module.save_space(task_idx, tnsr)

            target_r = max([r] + [module.saved_space[ttt][0].shape[1] for ttt in range(task_idx)])

            for ttt in range(task_idx + 1):
                # 对齐
                saved = module.saved_space[ttt][0]                
                
                if saved.shape[1] < target_r:
                    new = torch.zeros((768, target_r))
                    new[:, :saved.shape[1]] = saved
                    module.saved_space[ttt][0] = new
                
            module.reset_input_matrix()               

    @torch.no_grad()
    def _update_feature(self, task_idx, train_loader, test_loaders):
        '''
        Update feature lists and the corresponding type
        '''

        self._update_input_matrix(train_loader, test_loaders[0].dataset.trfms)

        threshold = (self.lame - self.lamb)*task_idx/self.task_num + self.lamb

        if task_idx == 0:
            for i, attention_module in enumerate(self.attention_modules):
                activation = attention_module.cur_matrix

                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_ratio = (S**2)/(S**2).sum()
                r = max(np.sum(np.cumsum(sval_ratio) < threshold), 1)
                assert r < activation.shape[0]/2

                self.feature_list_each_tasks[task_idx][i] = U[:, :r]
                self.feature_list.append(U[:, :r])
                self.project_type.append('remove')

                attention_module.reset_input_matrix()                
        else:
            for i, attention_module in enumerate(self.attention_modules):

                activation = attention_module.cur_matrix
                _, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()

                if self.project_type[i] == 'remove':

                    act_hat = activation - torch.Tensor(self.feature_list[i] @ self.feature_list[i].transpose()) @ activation
                    U, S, _ = np.linalg.svd(act_hat, full_matrices = False)
                    sigma = S**2

                    delta = (torch.tensor(self.feature_list[i]).T @ activation @ activation.T @ torch.tensor(self.feature_list[i])).diagonal()

                    stack = np.hstack((delta, sigma))
                    stack_index = np.argsort(stack)[::-1] # the index of each element in descending sorted array
                    stack = np.sort(stack)[::-1] # descending sorted array

                    if threshold * sval_total <= 0:
                        r = 0
                    else:
                        r = min(np.sum(np.cumsum(stack) < threshold * sval_total) + 1, activation.shape[0])

                    Ui = np.hstack((self.feature_list[i], U))
                    sel_each = stack_index[:r]
                    sel_overall = sel_each[sel_each >= len(delta)] # without overlap

                    self.feature_list[i] = np.hstack((self.feature_list[i], Ui[:, sel_overall]))
                    self.feature_list_each_tasks[task_idx][i] = Ui[:, sel_each]

                    if sel_overall.shape[0] == 0:
                        print(f'Skip Updating Space for layer: {i+1}')

                else:
                    act_hat = Torch.Tensor(self.feature_list[i] @ self.feature_list[i].transpose()) @ activation
                    U,S,_ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total     
                    accumulated_sval = sval_hat/sval_total          

                    if accumulated_sval < 1 - threshold:
                        print (f'Skip Updating Space for layer: {i+1}')
                    else:
                        r = np.sum(accumulated_sval - np.cumsum(sval_ratio) >= 1 - threshold) + 1
                        act_feature = self.feature_list[i] - U[:,0:r] @ U[:,0:r].T @ self.feature_list[i]
                        U, _, _ = np.linalg.svd(act_feature)
                        self.feature_list[i]=U[:,:self.feature_list[i].shape[1]-r]

                attention_module.reset_input_matrix()

        print('-'*40)
        print(f'Threshold: {threshold}')
        print('-'*40)
        for i in range(len(self.feature_list)):
            '''
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            '''
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

    @torch.no_grad()
    def _update_input_matrix(self, train_loader, test_trfms):

        if self.eval_mat:
            self._network.eval()
            train_trfms = train_loader.dataset.trfms
            train_loader.dataset.trfms = test_trfms

        for batch in tqdm(train_loader, desc = "Forwarding to get input matrix"):
            self._network.update_input_matrix(batch['image'].to(self.device))

        if self.eval_mat:
            self._network.train()
            train_loader.dataset.trfms = train_trfms

    def get_parameters(self, config):
        return self._network.parameters()        