"""
@inproceedings{10.24963/ijcai.2024/456,
  author = {Hong, Chenxing and Jin, Yan and Kang, Zhiqi and Chen, Yizhou and Li, Mengke and Lu, Yang and Wang, Hanzi},
  title = {Dynamically anchored prompting for task-imbalanced continual learning},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  year = {2025},
}
https://dl.acm.org/doi/10.24963/ijcai.2024/456
Adapted from https://github.com/chenxing6666/dap
"""

import math
import copy
import torch
import torch.nn.functional as F
from .finetune import Finetune
import numpy as np
from torch.utils.data import DataLoader

global_max_dist = torch.tensor(0)
global_max_dist2 = torch.tensor(0)
global_lam = 0.25


class DAP(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = backbone
        self.train_mask = kwargs['train_mask']
        self.task_inc = kwargs['task_inc']
        self.pull_constraint = kwargs['pull_constraint']
        self.pull_constraint_coeff = kwargs['pull_constraint_coeff']

        self.task_idx = 0
        self.task_data_count = []
        self.prompt_center = None

        # initialize class_mask
        if self.num_class % kwargs['task_num'] != 0:
            raise ValueError('Number of classes must be divisible by number of tasks')
        classes_per_task = self.num_class // kwargs['task_num']
        self.class_mask = [list(range(i * classes_per_task, (i + 1) * classes_per_task)) for i in range(kwargs['task_num'])]

        self.original_model = copy.deepcopy(self.backbone)
        self.original_model.to(self.device)
        self.original_model.eval()

        if kwargs['freeze']:
            # all parameters are frozen for original vit model
            for p in self.original_model.parameters():
                p.requires_grad = False

            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self.network.named_parameters():
                if n.startswith(tuple(kwargs['freeze'])):
                    p.requires_grad = False

        self.loss_fn.to(self.device)

    def observe(self, data, train_gprompt=False, gen=False):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            if self.original_model is not None:
                output = self.original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        if gen:
            output = self.network(x, task_id=self.task_idx, cls_features=cls_features, train=True, gen=gen)
        else:
            output = self.network(x, task_id=self.task_idx, cls_features=cls_features, train=True)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if self.train_mask and self.class_mask is not None:
            mask = self.class_mask[self.task_idx]
            not_mask = np.setdiff1d(np.arange(self.num_class), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(
                dim=1, index=not_mask, value=float('-inf'))

        if (train_gprompt):

            pla_similarity_loss_res = self.cal_latestsimilarity_loss(
                model=self.network, task_id=self.task_idx)
            sta_similarity_loss_res = self.cal_similarity_loss(model=self.network, task_id=self.task_idx, prompt_center=self.prompt_center)

            pla_similarity_loss = pla_similarity_loss_res['similarity']
            sta_similarity_loss = sta_similarity_loss_res['avg_similarity']

            min_data_count = min(self.task_data_count)
            max_data_count = max(self.task_data_count)
            last_data_count = self.task_data_count[-1]
            epsilon = 1e-10
            alpha = (last_data_count - min_data_count) / (max_data_count - min_data_count + epsilon)

            loss2 = alpha*sta_similarity_loss
            loss3 = (1-alpha)*pla_similarity_loss

            loss = self.loss_fn(logits, y) + loss2 + loss3

        else:
            # base criterion (CrossEntropyLoss)
            loss = self.loss_fn(logits, y)
        if self.pull_constraint and 'reduce_sim' in output:
            loss = loss - self.pull_constraint_coeff * output['reduce_sim']

        if not math.isfinite(loss.item()):
            raise RuntimeError(f'Loss is {loss.item()}, stopping training')

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            if self.original_model is not None:
                output = self.original_model(x)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        output = self.network(x, task_id=self.task_idx, cls_features=cls_features, gen=True)
        logits = output['logits']

        # adding mask to output logits
        if self.task_inc and self.class_mask is not None:
            mask = self.class_mask[self.task_idx]
            mask = torch.tensor(mask, dtype=torch.int64).to(self.device)
            logits_mask = torch.ones_like(logits, device=self.device) * float('-inf')
            logits_mask = logits_mask.index_fill(1, mask, 0.0)
            logits = logits + logits_mask

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.network.task_id = task_idx
        self.task_data_count.append(len(train_loader.dataset))

    @staticmethod
    def cal_latestsimilarity_loss(model: torch.nn.Module, task_id=-1):
        res = dict()
        global global_max_dist2

        gprompt = model.prompt.generalprompt
        tprompt = model.prompt.taskprompt[task_id].detach()

        gprompt_flat = gprompt.view(-1)
        tprompt_tensors = tprompt.view(-1)
        similarity = 1-F.cosine_similarity(gprompt_flat, tprompt_tensors, dim=0)
        res['similarity'] = similarity
        return res

    @staticmethod
    def cal_center(model: torch.nn.Module, task_id=-1, task_data_count=None, prompt_center=None):
        tprompt = model.prompt.taskprompt
        if task_id > 0:
            if prompt_center is None:
                prompt_center = tprompt[0].detach().view(-1)
            current_tprompt = tprompt[task_id - 1].detach().view(-1)
            if task_data_count:
                weights = [1 / count for count in task_data_count[:task_id]]
                normalized_weight = weights[-1] / sum(weights)
                weights2 = sum(weights[:-1]) / sum(weights)
            else:
                normalized_weight = 1.0 / task_id
            prompt_center = (prompt_center * weights2) + \
                (current_tprompt * normalized_weight)
        else:
            prompt_center = torch.zeros_like(tprompt[0].detach().view(-1))
        return prompt_center

    @staticmethod
    def cal_similarity_loss(model: torch.nn.Module, task_id=-1, prompt_center=None):
        res = dict()
        global global_max_dist

        gprompt = model.prompt.generalprompt

        if task_id > 0:
            gprompt_flat = gprompt.view(-1)
            similarity = 1-F.cosine_similarity(gprompt_flat, prompt_center, dim=0)
            res['similarity'] = similarity
            res['avg_similarity'] = similarity
        else:
            res['similarity'] = torch.tensor(0)
            res['avg_similarity'] = 0
        return res