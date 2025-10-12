# -*- coding: utf-8 -*-
"""
@article{jiao2024vector,
  title={Vector Quantization Prompting for Continual Learning},
  author={Jiao, Li and Lai, Qiuxia and Li, Yu and Xu, Qiang},
  journal={NeurIPS},
  year={2024}
}
"""

import math
import copy
import sys
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from core.scheduler import CosineSchedulerIter
from core.utils import logger
from .finetune import Finetune
from core.model.backbone.resnet import *
import numpy as np
from torch.utils.data import DataLoader, Dataset
from timm.utils import accuracy
from torch.distributions.multivariate_normal import MultivariateNormal


class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        
    def forward(self, x, train=True):
        if train:
            feat, loss = self.backbone(x, train=True)
            return self.classifier(feat), loss
        else:
            feat = self.backbone(x, train=False)
            return self.classifier(feat)
    
    def forward_fc(self, x):
        # x = self.backbone.norm(x)
        out = self.classifier(x)
        return out


class VQPrompt(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = Model(self.backbone, feat_dim, kwargs['init_cls_num'])
        # --- KEY CHANGE IS HERE ---
        # 1. Specify 'vq' as the prompt type.
        # 2. Pass VQPrompt's required parameters: pool_size, prompt_length, and temperature (soft_t).
        #    We assume 'temperature' is provided in kwargs instead of 'mu'.
        self.network.backbone.create_prompt(
            'qt', 
            n_tasks = kwargs['task_num'], 
            prompt_param=[kwargs['pool_size'], kwargs['prompt_length'], kwargs['temperature']]
        )
        # --- END OF KEY CHANGE ---
        self.task_idx = 0
        self.kwargs = kwargs
        
        self.last_out_dim = 0

        # --- new config ---
        self.crct_epochs = kwargs['crct_epochs']
        self.n_centroids = kwargs['n_centroids']
        self.adaptive_pred = kwargs['adaptive_pred']
        self.ca_lr = kwargs['ca_lr']
        self.ca_weight_decay = kwargs['ca_weight_decay']
        self.ca_batch_size_ratio = kwargs['ca_batch_size_ratio']
        self.batch_size = kwargs['batch_size']  

        # --- lr schedule config ---
        self.base_value =  1e-5                   #kwargs['lr_scheduler']['kwargs']['base_value']
        self.final_value = 1e-6                          #kwargs['lr_scheduler']['kwargs']['final_value']
        self.iter_step =  40                #kwargs['lr_scheduler']['kwargs']['iter_step']
        self.n_epochs = 10                #kwargs['lr_scheduler']['kwargs']['n_epochs']

        # 初始化均值和协方差存储
        self.network.cls_mean = {}
        self.network.cls_cov = {}


    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        """
        This function's logic is about managing the classifier and task state,
        which is independent of the prompt type. No changes needed.
        """
        self.task_idx = task_idx
        self.network.backbone.task_id = task_idx
        
        in_features = self.network.classifier.in_features
        out_features = self.network.classifier.out_features
        new_out_features = self.kwargs['init_cls_num'] + task_idx * self.kwargs['inc_cls_num']
        new_fc = nn.Linear(in_features, new_out_features)
        new_fc.weight.data[:out_features] = self.network.classifier.weight.data
        new_fc.bias.data[:out_features] = self.network.classifier.bias.data
        self.network.classifier = new_fc
        self.network.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        self.out_dim = new_out_features
        self.dw_k = torch.tensor(np.ones(self.out_dim + 1, dtype=np.float32)).to(self.device)

    def observe(self, data):
        """
        The training step logic is also generic. The `loss` variable will
        automatically contain the VQ-loss from the prompt module. No changes needed.
        """
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit, loss = self.network(x, train=True)

        logit[:,:self.last_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(y.size()).long()]

        loss += (self.loss_fn(logit, y) * dw_cls).mean()
        
        pred = torch.argmax(logit, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss
    
        
    def after_task(self, task_idx,train_loader):
        self.last_out_dim = self.out_dim
        
        current_task_loader = train_loader.get_loader(task_idx)
        current_class_mask = train_loader.class_mask[task_idx]
        
        self._compute_mean(model=self.network ,train_loader=current_task_loader ,class_mask=current_class_mask)

        # pseudo replay
        if task_idx > 0: #实际是大于0，为了方便观察改一下
            self.train_task_adaptive_prediction(model=self.network,train_loader=current_task_loader ,class_mask=train_loader.class_mask , task_id=task_idx)


    def inference(self, data):
        """
        Inference logic is independent of the prompt type. No changes needed.
        """
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logit = self.network(x, train=False)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)


    def get_parameters(self, config):
        """
        Specifies which parameters to train. Same as CodaPrompt: only the prompt
        and the classifier head are trainable. No changes needed.
        """
        return list(self.network.backbone.prompt.parameters()) + list(self.network.classifier.parameters())

    @torch.no_grad() #新加的
    def _compute_mean(self,model: torch.nn.Module,train_loader, class_mask=None):
        # model.eval()

        for cls_id in class_mask:
            # train_loader.load_class(cls_id)
            # data_loader_cls = DataLoader(train_loader, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers) 

            # 创建单类别数据集
            single_class_dataset = self._create_single_class_dataset(train_loader, cls_id)
            
            # 创建单类别数据加载器
            data_loader_cls = DataLoader(
                single_class_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False, 
                num_workers=0  # 单线程避免问题
            ) 

            features_per_cls = []
            for batch_data in data_loader_cls:
                inputs = batch_data["image"]
                targets = batch_data["label"]
                # send data to gpu
                inputs = inputs.cuda()
                targets = targets.cuda()
                features = model.backbone(inputs)
                features_per_cls.append(features)
            features_per_cls = torch.cat(features_per_cls, dim=0)


            from sklearn.cluster import KMeans
            n_clusters = self.n_centroids  # default 10
            features_per_cls_np = features_per_cls.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
            kmeans.fit(features_per_cls_np)
            cluster_labels = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
                cluster_data = features_per_cls[cluster_labels == i]
                # cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).cuda()
                cluster_mean = cluster_data.mean(dim=0).to(dtype=torch.float64, device=self.device)
                # cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).cuda()
                cluster_var = cluster_data.var(dim=0).to(dtype=torch.float64, device=self.device)
                cluster_means.append(cluster_mean)
                cluster_vars.append(cluster_var)
            
            model.cls_mean[cls_id] = cluster_means
            model.cls_cov[cls_id] = cluster_vars

    def train_task_adaptive_prediction(self, model: torch.nn.Module,train_loader ,class_mask=None, task_id=-1): #新加的
        model.train()
        run_epochs = self.crct_epochs
        crct_num = 0
        valid_out_dim = self.last_out_dim
        ca_lr = self.ca_lr 
        weight_decay = self.ca_weight_decay
        batch_size = self.batch_size
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
        network_params = [{'params': param_list, 'lr': ca_lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.AdamW(network_params, lr=ca_lr / 10, weight_decay=weight_decay) 

        criterion = torch.nn.CrossEntropyLoss()
        if self.device == 'cuda':
            criterion = criterion.cuda()

        for i in range(task_id):  # only take part of the samples after random permute
            crct_num += len(class_mask[i])

        scheduler_cfg = {
                'base_value': [ca_lr / 10], 
                'final_value': [1e-6], 
                'optimizer': optimizer, 
                'iter_step': crct_num, 
                'n_epochs': run_epochs, 
                'last_epoch': -1, 
                'warmup_epochs': 0, 
                'start_warmup_value': 0, 
                'freeze_iters': 0
            }
        scheduler = CosineSchedulerIter(**scheduler_cfg)

        for epoch in range(run_epochs):

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = int(batch_size * self.ca_batch_size_ratio) # default 5

            metric_logger =logger.MetricLogger(delimiter="  ")
            metric_logger.add_meter('Lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('Loss', logger.SmoothedValue(window_size=1, fmt='{value:.4f}'))

            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mapped_c_id = c_id# mapped_c_id = class_map[c_id]

                    if c_id not in model.cls_mean or len(model.cls_mean[c_id]) == 0:
                        continue

                    for cluster in range(len(model.cls_mean[c_id])):
                        mean = model.cls_mean[c_id][cluster] #he root cause of your previous torch.cat error is that the mean vectors stored in model.cls_mean have different lengths for different classes/tasks.
                        var = model.cls_cov[c_id][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([mapped_c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
            sampled_label = torch.tensor(sampled_label).long().to(sampled_data.device)
            print(sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            # 随机打乱数据
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                try:
                    logits = model.module.forward_fc(inp)
                except:
                    logits = model.forward_fc(inp)

                logits = logits[:,:valid_out_dim]

                loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
                acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # step inside loop for Iter scheduler

                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            print("Averaged stats:", metric_logger)


    def _create_single_class_dataset(self, train_loader, cls_id):
        """创建只包含特定类别数据的数据集"""
        from torch.utils.data import Dataset
        
        class SingleClassDataset(Dataset):
            def __init__(self, original_loader, target_cls_id):
                self.data = []
                self.labels = []
                self.transform = original_loader.dataset.trfms
                
                # 从原始数据集中筛选特定类别的数据
                original_dataset = original_loader.dataset
                for i in range(len(original_dataset)):
                    data_item = original_dataset[i]
                    if data_item["label"] == target_cls_id:
                        self.data.append(data_item["image"])
                        self.labels.append(data_item["label"])
                        
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return {
                    "image": self.data[idx],
                    "label": self.labels[idx]
                }
        
        return SingleClassDataset(train_loader, cls_id)