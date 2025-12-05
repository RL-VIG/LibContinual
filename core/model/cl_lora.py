"""
@article{He_2025_CVPR,
    author    = {He, Jiangpeng and Duan, Zhihao and Zhu, Fengqing},
    title     = {CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning},
    journal = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {30534-30544}
}

Adapted from https://github.com/JiangpengHe/CL-LoRA
"""

import math
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from copy import deepcopy
from torch.nn import functional as F

from .backbone.transformer import MultiHeadAttention_CL_LoRA

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

def compute_orthogonality_loss(previous_weights_list, current_weights, epsilon=1e-8):
    total_ortho_loss = 0.0
    current_norm = torch.norm(current_weights.flatten())
    current_normalized = current_weights.flatten() / (current_norm + epsilon)

    for prev_weights in previous_weights_list:
        # Normalize previous weights
        prev_norm = torch.norm(prev_weights.flatten())
        prev_normalized = prev_weights.flatten() / (prev_norm + epsilon)

        # Compute absolute dot product (should be close to 0 for orthogonal vectors)
        dot_product = torch.abs(torch.sum(prev_normalized * current_normalized))

        total_ortho_loss += dot_product

    # Average over all previous tasks
    if len(previous_weights_list) > 0:
        total_ortho_loss /= len(previous_weights_list)

    return total_ortho_loss

class CosineLinearFeature(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinearFeature, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

    def forward_diagonal(self, input, cur_task, alpha=0., beta=0.0, init_cls=10, inc=10, out_dim=768, use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc
            input1 = F.normalize(input[:, i * out_dim:(i + 1) * out_dim], p=2, dim=1)
            weight1 = F.normalize(self.weight[start_cls:end_cls, i * out_dim:(i + 1) * out_dim], p=2, dim=1)

            out = F.linear(input1, weight1)
            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out

        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        return {'logits': out_all}

class Model(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.inc = kwargs["inc_cls_num"]
        self.init_cls = kwargs["init_cls_num"]
        self._cur_task = -1
        self.out_dim =  768
        self.fc = None
        self.alpha = 0.
        self.beta = 0
        self.fc_list = nn.ModuleList()
        self.fc_list_task = nn.ModuleList()
        self.adapter_list = nn.ModuleList()
        self.init_proto = None

        self._device = device

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    @property
    def feature_dim(self):

        return self.out_dim * (self._cur_task + 1)

    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        init_proto = self.generate_fc(self.out_dim, nb_classes).to(self._device)

        if self.init_proto is not None:
            old_nb_classes = self.init_proto.out_features
            weight = deepcopy(self.init_proto.weight.data)
            init_proto.weight.data[: old_nb_classes, :] = nn.Parameter(weight)
        del self.init_proto
        self.init_proto = init_proto

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)

        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        del self.proxy_fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinearFeature(in_dim, out_dim)
        return fc
    
    def forward_kd(self, x, t_idx):
        x_new, x_teacher = self.backbone.forward_general_cls(x, t_idx)
        out_new, out_teacher = self.proxy_fc(x_new), self.proxy_fc(x_teacher)
        return  out_new, out_teacher

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, test=False)
            out = self.proxy_fc(x)
            out.update({"features": x})
            return out
        else:

            x_input = self.backbone.forward(x, test=True)
            out = self.fc.forward_diagonal(x_input, cur_task=self._cur_task, alpha=0., init_cls=self.init_cls, inc=self.inc, use_init_ptm=False, beta=0)
            out.update({"features": x_input})

            return out

class CL_LoRA(nn.Module):

    def __init__(self, backbone, device, **kwargs):

        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = 0

        self._network = Model(backbone, device, **kwargs)
        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_CL_LoRA)]

        self.lora_modules = [[] for _ in range(self.task_num)]
        self.lora_scale_weights = [[] for _ in range(self.task_num)]
        self.optim = None

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        aux_targets = y - self._known_classes

        logits = self._network(x, test=False)['logits']
        loss = F.cross_entropy(logits, aux_targets)

        if self._cur_task > 0:
            
            kd_ratio = 5.
            Temperature = 2

            out_new, out_teacher = self._network.forward_kd(x, self._cur_task)
            out_new_logits = out_new["logits"]
            out_teacher_logits = out_teacher["logits"]
            loss_kd = kd_ratio * _KD_loss(out_new_logits, out_teacher_logits, T=Temperature)

            self.optim.zero_grad()
            loss_kd.backward()

            for j in range(len(self._network.backbone.feat.general_pos)):
                pos = self._network.backbone.feat.adapt_pos.index(self._network.backbone.feat.general_pos[j])
                for jj in range(len(self._network.backbone.feat.msa)):
                    if self._network.backbone.feat.msa[jj] == 1:
                        temp_weights = 1. * torch.norm(self._network.backbone.feat.old_adapter_list[self._cur_task-1][pos][jj].lora_A.weight,dim=1)
                        temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                        self._network.backbone.feat.cur_adapter[pos][jj].lora_A.weight.grad = temp_weights.unsqueeze(1) * self._network.backbone.feat.cur_adapter[pos][jj].lora_A.weight.grad

            self.optim.step()

            orth_loss_specific = compute_orthogonality_loss(self._network.backbone.feat.block_weight_list, self._network.backbone.feat.block_weight)
            loss += 0.0001 * orth_loss_specific

        preds = logits.max(1)[1]
        correct_count = preds.eq(aux_targets).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc, loss
    
    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self._network(x, True)["logits"]
        preds = logits.max(1)[1]

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx > 0:
            self._known_classes = self._total_classes
            self._network.freeze()
            self._network.backbone.add_adapter_to_list()

        self._cur_task = task_idx
        self._total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self._network.update_fc(self._total_classes)

        for name, param in self._network.named_parameters():
            if 'backbone.feat.cur_adapter' in name or 'proxy_fc.' in name or 'init_proto' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

            param.requires_grad_(False)

            if 'lora' in name and 'cur_adapter' in name:
                if any(f'er.{i}.' in name for i in range(6)) and 'lora_B' in name and 'cur_adapter':
                    pass
                else:
                    param.requires_grad_(True)

            elif f'proxy_fc' in name:
                param.requires_grad_(True)
            elif 'init_proto' in name:
                param.requires_grad_(True)
            elif 'block_weight' in name and 'old' not in name: 
                param.requires_grad_(True)

        self._network = self._network.to(self.device)

    @torch.no_grad()
    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._network.add_fc()
        train_loader.dataset.trfms = test_loaders[0].dataset.trfms
        self.replace_fc(train_loader)

        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            for index in range(0, self._cur_task + 1):
                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    data, label = batch['image'], batch['label']
                    data = data.to(self.device)
                    label = label.to(self.device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(train_loader.dataset.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

    def get_parameters(self, config):
        return self._network.parameters()

    def set_optim(self, optim):
        self.optim = optim