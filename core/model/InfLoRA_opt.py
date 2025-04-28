"""
@inproceedings{liang2024inflora,
    title={InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning},
    author={Liang, Yan-Shuo and Li, Wu-Jun},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={23638--23647},
    year={2024}
}

Adapted from https://github.com/liangyanshuo/InfLoRA
"""

import os
import math
import torch
import random
import torch.nn as nn
import numpy as np

from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from .backbone.transformer import MultiHeadAttention_LoRA, VisionTransformer
from .backbone.clip import CLIP, tokenize
from .backbone.vit import ViTZoo

VIT = ViTZoo
CLIP = CLIP

def _set_random(seed):
    '''
    Set random values on various devices to ensure repeatable results
    '''

    seed = int(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SiNet(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.device = device

        if isinstance(backbone, VIT):
            _set_random(os.environ["PYTHONHASHSEED"])
            self.classifier_pool = nn.ModuleList([
                nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
                [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)]
            )
        elif isinstance(backbone, CLIP):
            self.accm_class_names = []   
            self.curr_class_names = []
            self.accm_text_tokens = None
            self.curr_text_tokens = None

            self.prompt_template = kwargs['prompt_template']
        else:
            assert 0, f'Backbone not implemented'

    def update_fc(self, train_loader):
        
        self._cur_task_id += 1

        if isinstance(self.backbone, CLIP):

            self.curr_class_names = train_loader.dataset.get_class_names()
            self.accm_class_names += self.curr_class_names

            self.curr_text_tokens = tokenize(
                [self.prompt_template.format(c) for c in self.curr_class_names]
            ).to(self.device)

            self.accm_text_tokens = tokenize(
                [self.prompt_template.format(c) for c in self.accm_class_names]
            ).to(self.device)
    
    # These two for classifier alignment, 
    def get_feature(self, x):
        if isinstance(self.backbone, VIT):
            return self.backbone(x)
        elif isinstance(self.backbone, CLIP):
            assert 0
        else:
            assert 0
        
    def fc_only(self, x):
        if isinstance(self.backbone, VIT):
            logits = []
            for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                logits.append(prompts(x))
            return torch.cat(logits, dim=1)
        elif isinstance(self.backbone, CLIP):
            assert 0
        else:
            assert 0
        
    def forward(self, x, inference = False):

        if isinstance(self.backbone, VIT):
            
            logits = []
            features = self.backbone(x)

            if inference:
                for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                    logits.append(prompts(features))
            else:
                for prompts in [self.classifier_pool[self._cur_task_id]]:
                    logits.append(prompts(features))

            return torch.cat(logits, dim=1)

        elif isinstance(self.backbone, CLIP):
            if inference:
                features_img, features_txt, logits_per_img, logits_per_txt = self.backbone(x, self.accm_text_tokens)
            else:
                features_img, features_txt, logits_per_img, logits_per_txt = self.backbone(x, self.curr_text_tokens)
            return logits_per_img
        else:
            assert 0, f'Backbone not implemented'

    def update_input_matrix(self, x):
        
        if isinstance(self.backbone, VIT):
            self.backbone(x, get_input_matrix = True)

        elif isinstance(self.backbone, CLIP):
            self.backbone(image = x, text = self.curr_text_tokens, get_input_matrix = True)

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

        self._dataset = kwargs['dataset']
        self._use_class_alignment = kwargs['use_ca']
        self._logit_norm = None if self._dataset == 'cifar100' else 0.1
        self._class_means = None
        self._class_covs = None

        self._network = SiNet(backbone, device, **kwargs).to(self.device)

        if isinstance(backbone, VIT):
            self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_LoRA)]
        elif isinstance(backbone, CLIP):
            self.visual_only = kwargs['visual_only']
            if self.visual_only:
                self.attention_modules = [module for name, module in self._network.named_modules() if isinstance(module, MultiHeadAttention_LoRA) and 'visual' in name]
            else:
                self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_LoRA)]
        else:
            assert 0, 'Not Implmented'

    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.
        '''
        
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
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        It is called before the training of each task to update the parameters, select the branch for training, and update the lora_A matrix of the corresponding branch
        '''

        if task_idx == 1:
            self._known_classes = self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num
        self._network.update_fc(train_loader)

        _set_random(os.environ["PYTHONHASHSEED"])
        for module in self.attention_modules:
            module.init_param()

        unfrezeed_params = []
        if isinstance(self._network.backbone, VIT):
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if f"classifier_pool.{task_idx}." in name or "lora_B" in name:
                    param.requires_grad_(True)
                    unfrezeed_params.append(name)
        elif isinstance(self._network.backbone, CLIP):
            if self.visual_only:
                for name, param in self._network.named_parameters():
                    param.requires_grad_(False)
                    if "visual" in name and "lora_B" in name:
                        param.requires_grad_(True)
                        unfrezeed_params.append(name)
            else:
                for name, param in self._network.named_parameters():
                    param.requires_grad_(False)
                    if "lora_B" in name:
                        param.requires_grad_(True)
                        unfrezeed_params.append(name)

        print(f"Current task : {task_idx}, Parameters to be updated: {len(unfrezeed_params)}")
        print(",\n".join(unfrezeed_params))

        _set_random(os.environ["PYTHONHASHSEED"])
        for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):
            self._network.update_input_matrix(x = batch['image'].to(self.device))


        if task_idx == 0:
            for module in self.attention_modules:
                assert module.n_cur_matrix > 0
                U, S, _ = torch.linalg.svd(module.cur_matrix, full_matrices=False)

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

                U, _, _ = torch.linalg.svd(cur_matrix, full_matrices = False)
                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.reset_input_matrix()
    
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        for module in self.attention_modules:
            module.merge_weight()

        self._update_feature(task_idx, train_loader, test_loaders[0].dataset.trfms)
        if self._use_class_alignment:
            self._create_distribution(train_loader, test_loaders[0].dataset.trfms)
            if task_idx > 0:
                self._compact_classifier(task_idx)

    @torch.no_grad()
    def _update_feature(self, task_idx, train_loader, test_trfms):
        '''
        Update feature lists and the corresponding type
        '''

        _set_random(os.environ["PYTHONHASHSEED"])
        for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):

            self._network.update_input_matrix(x = batch['image'].to(self.device))

        threshold = (self.lame - self.lamb)*task_idx/self.task_num + self.lamb

        if task_idx == 0:
            for i, attention_module in enumerate(self.attention_modules):
                activation = attention_module.cur_matrix

                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = max(np.sum(np.cumsum(sval_ratio) < threshold), 1)
                assert r < activation.shape[0]/2

                self.feature_list.append(U[:, :r])
                self.project_type.append('remove')

                attention_module.reset_input_matrix()                
        else:
            for i, attention_module in enumerate(self.attention_modules):

                activation = attention_module.cur_matrix
                _, S, _ = np.linalg.svd(activation, full_matrices=False)
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

    @torch.no_grad()
    def _create_distribution(self, train_loader, test_trfms):
        
        self._network.eval()
        train_loader.dataset.trfms = test_trfms

        samples = [[] for _ in range(self.inc_cls_num)]
        for batch in train_loader:
            x, y = batch['image'], batch['label'] - self._known_classes
            for label in range(self.inc_cls_num):
                samples[label].append(x[y == label])
        samples = [torch.cat(label_sample, dim = 0).to(self.device) for label_sample in samples]

        # Computing class mean
        if self._class_means is None:
            self._class_means = torch.zeros((self.init_cls_num, 768))
            self._class_covs = torch.zeros((self.init_cls_num, 768, 768))
        else:
            self._class_means = torch.cat((self._class_means, torch.zeros((self.inc_cls_num, 768))), dim=0)
            self._class_covs = torch.cat((self._class_covs, torch.zeros((self.inc_cls_num, 768, 768))), dim=0)

        for class_idx, x in enumerate(samples):
            class_idx += self._known_classes
            features = self._network.get_feature(x)

            self._class_means[class_idx, :] = torch.mean(features, dim = 0)
            self._class_covs[class_idx, :, :] = torch.cov(features.to(torch.float64).T) + torch.eye(768, device = self.device) * 1e-4

    def _compact_classifier(self, task_idx):

        # Hyperparam
        epoch = 5
        lr = 0.01
        weight_decay = 0.0005
        momentum = 0.9
        num_sample = 256

        for param in self._network.classifier_pool[:task_idx + 1].parameters():
            param.requires_grad_(True)
        param_list = [param for param in self._network.classifier_pool.parameters() if param.requires_grad]

        optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)

        for ep in range(epoch):
            sampled_data, sampled_label = [], []

            for class_id in range((task_idx + 1) * self.inc_cls_num):
                task_id = class_id // self.inc_cls_num

                decay = (task_id + 1) / (task_idx + 1) * 0.1
                cls_mean = self._class_means[class_id].to(self.device, torch.float64) * (0.9 + decay)
                cls_cov = self._class_covs[class_id].to(self.device)

                m = torch.distributions.multivariate_normal.MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sample,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([class_id] * num_sample)

            inputs = torch.cat(sampled_data, dim=0).float().to(self.device)
            targets = torch.tensor(sampled_label).long().to(self.device)

            # Randomize
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            
            for _iter in range((task_idx + 1) * self.inc_cls_num):
                
                inp = inputs[_iter * num_sample : (_iter+1) * num_sample]
                tgt = targets[_iter * num_sample : (_iter+1) * num_sample]
                logits = self._network.fc_only(inp)

                if self._logit_norm:

                    pass

                else:
                    loss = F.cross_entropy(logits, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

    def get_parameters(self, config):
        return self._network.parameters()

