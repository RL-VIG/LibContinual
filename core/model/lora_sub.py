"""
@inproceedings{liu2025lora,
  title={LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

Adapted from https://github.com/scarlet0703/LoRA-Sub-DRS
"""

import torch
import torch.nn as nn
import copy
import numpy as np
import math

from copy import deepcopy
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist

from .backbone.transformer import MultiHeadAttention_LoRA_Sub

class AugmentedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, norm=2):
        super(AugmentedTripletLoss, self).__init__()
        self.margin = margin
        self.norm = norm
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, center):
        device = (torch.device('cuda')
                  if inputs.is_cuda
                  else torch.device('cpu'))
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)

        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        num_proto = len(center)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if dist[i][mask[i] == 0].numel() == 0:
                dist_an.append((dist[i][mask[i]].max()+self.margin).unsqueeze(0))
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        if num_proto > 0:
            center = torch.from_numpy(center / np.linalg.norm(center, axis=1)[:, None]).to(device)
            for i in range(n):
                for j in range(num_proto):
                    distp = torch.norm(inputs[i].unsqueeze(0) - center[j], self.norm).clamp(min=1e-12)
                    dist_an[i] = min(dist_an[i].squeeze(0), distp).unsqueeze(0)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False, thres=1.001,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svd=svd,
                        thres=thres)
        super(Adam, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)

                if svd and len(self.transforms) > 0:
                    if len(update.shape) == 4:
                        # the transpose of the manuscript
                        update_ = torch.mm(update.view(update.size(
                            0), -1), self.transforms[p]).view_as(update)
                    else:
                        if self.transforms[p].shape[0]==update.shape[0]:
                            update_ = torch.mm(self.transforms[p], update)
                        else:
                            update_ = torch.mm(update, self.transforms[p])
                else:
                    update_ = update

                p.data.add_(update_)
        return loss

    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue

            for p in group['params']:
                thres = group['thres']
                if p.requires_grad == False or thres == 1.0:
                    continue
                eigen_values = self.eigens[p]['eigen_value']
                cumulative_sum = eigen_values.cumsum(dim=0) / eigen_values.sum()
                num_vectors = (cumulative_sum >= thres).nonzero(as_tuple=True)[0][0] + 1
                print('reserving basis {}/{}; cond: {}, ratio:{}'.format(
                    num_vectors, eigen_values.shape[0],
                    eigen_values[0] / eigen_values[-1],
                    cumulative_sum[num_vectors - 1]
                ))
                basis = self.eigens[p]['eigen_vector'][:, :num_vectors]
                transform = torch.mm(basis, basis.transpose(1, 0))
                self.transforms[p] = transform / torch.norm(transform)
                self.transforms[p].detach_()

    def get_eigens(self, fea_in):

        for group in self.param_groups:
            if group['svd']:
                for p in group['params']:
                    if p.requires_grad:
                        eigen = self.eigens[p]
                        _, eigen_value, eigen_vector = torch.svd(fea_in[p], some=False)
                        eigen['eigen_value'] = eigen_value
                        eigen['eigen_vector'] = eigen_vector

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update

class Model(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.device = device
        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)]
        )

    def update_fc(self):
        
        self._cur_task_id += 1

    def update_input_matrix(self, x):
        
        self.backbone(x, get_input_matrix = True)

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):

        logits = []
        features = self.backbone(x)

        for prompts in [self.classifier_pool[self._cur_task_id]]:
            logits.append(prompts(features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': features
        }

class LoRAsub_DRS(nn.Module):

    def __init__(self, backbone, device, **kwargs):

        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.fc_lrate = kwargs["fc_lrate"]
        self.margin_inter = kwargs["margin_inter"]
        self.lambada = kwargs["lambada"]
        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = 0

        self._network = Model(backbone, device, **kwargs)
        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_LoRA_Sub)]
        self.criterion = AugmentedTripletLoss(margin=self.margin_inter).to(self.device)
        self._protos = []

    def observe(self, data):
        
        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        outputs = self._network(x)
        logits, features = outputs['logits'], outputs['features']
        
        ATL = self.criterion(
            features / features.norm(dim=-1, keepdim=True),
            y,
            self._protos
        )
        loss = F.cross_entropy(logits, y) + self.lambada * ATL

        preds = logits.max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc, loss
    
    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        features = self._network.extract_features(x)
        features = (features.T / (np.linalg.norm(features.T, axis=0) + 1e-8)).T

        class_means = self._protos / np.linalg.norm(self._protos, axis=1)[:, None]

        dists = cdist(class_means, features, 'sqeuclidean')
        scores = dists.T

        #preds = np.argsort(scores, axis=1)[:, :1]
        preds = np.argmin(scores, axis=1) 

        correct_count = (preds == y.cpu().numpy()).sum()
        acc = correct_count / y.size(0)

        return preds, acc
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self._known_classes = self._total_classes
        self._total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

        self._network.update_fc()
        self._network = self._network.to(self.device)

        for module in self.attention_modules:
            module.init_param()

        unfrezeed_params = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if f'classifier_pool.{self._cur_task}.' in name or \
               f'lora'in name:
                param.requires_grad_(True)
                unfrezeed_params.append(name)

        print(f"Current task : {task_idx}, Parameters to be updated: {len(unfrezeed_params)}")

        if task_idx > 0:
            for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):
                self._network.update_input_matrix(x = batch['image'].to(self.device))

            self.fea_in  = {}

            for module in self.attention_modules:
                self.fea_in[module.lora_A_k.weight] = deepcopy(module.cur_matrix).to(self.device)
                self.fea_in[module.lora_A_v.weight] = deepcopy(module.cur_matrix).to(self.device)
                self.fea_in[module.lora_B_k.weight] = deepcopy(module.cur_matrix).to(self.device)
                self.fea_in[module.lora_B_v.weight] = deepcopy(module.cur_matrix).to(self.device)
                module.reset_input_matrix()

    @torch.no_grad()
    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        for module in self.attention_modules:
            module.save_weight()

        # Build Proto
        for class_idx in range(self._known_classes, self._total_classes):

            inputs_list = []

            for batch in train_loader:
                x, y = batch['image'].to(self.device), batch['label'].to(self.device)
                inputs_list.append(x[y == class_idx])

            class_inputs = torch.cat(inputs_list, dim=0)
            features_list = []

            for start_idx in range(0, class_inputs.shape[0], 128):
                end_idx = min(start_idx + 128, class_inputs.shape[0])
                batch_inputs = class_inputs[start_idx:end_idx].to(self.device)
                feats = self._network.extract_features(batch_inputs)
                features_list.append(feats.detach().cpu().numpy())

            features = np.concatenate(features_list, axis=0)
            class_mean = np.mean(features, axis=0)
            self._protos.append(class_mean)

        assert len(self._protos) > 0

        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self._cur_task += 1

    def get_parameters(self, config):
        return self._network.parameters()

    def get_optimizer(self, lr, weight_decay):

        fea_params = []
        for module in self.attention_modules:
            fea_params.append(module.lora_A_k.weight)
            fea_params.append(module.lora_A_v.weight)
            fea_params.append(module.lora_B_k.weight)
            fea_params.append(module.lora_B_v.weight)

        cls_params = [
            self._network.classifier_pool[self._cur_task].weight, 
            self._network.classifier_pool[self._cur_task].bias, 
        ]

        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': lr,
                                           'thres': 0.99},
                                          {'params': cls_params, 'weight_decay': weight_decay,
                                           'lr': self.fc_lrate}],
                               'weight_decay': weight_decay,
                               'betas': (0.9, 0.999)
                               }

        optim = Adam(**model_optimizer_arg)

        if self._cur_task > 0:
            optim.get_eigens(self.fea_in)
            optim.get_transforms()
            
        return optim