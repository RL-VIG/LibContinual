# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/HouPLWL19,
  title        = {Learning a Unified Classifier Incrementally via Rebalancing},
  author       = {Saihui Hou and
                  Xinyu Pan and
                  Chen Change Loy and
                  Zilei Wang and
                  Dahua Lin},
  booktitle    = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
                  2019, Long Beach, CA, USA, June 16-20, 2019},
  pages        = {831--839},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2019}
}
https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html

Adapted from https://github.com/hshustc/CVPR19_Incremental_Learning
"""

import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune
from core.model.backbone.resnet import *
import numpy as np
from torch.utils.data import DataLoader


cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs



class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = CosineLinear(feat_dim, num_class)
        
    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        logits = self.classifier(self.backbone(x)['features'])
        return logits



class LUCIR(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.network = Model(self.backbone, feat_dim, kwargs['init_cls_num'])
        self.K = kwargs['K']
        self.lw_mr = kwargs['lw_mr']
        self.ref_model = None
        self.task_idx = 0

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx

        if task_idx == 1:
            self.ref_model = copy.deepcopy(self.network)
            in_features = self.network.classifier.in_features
            out_features = self.network.classifier.out_features
            new_fc = SplitCosineLinear(in_features, out_features, self.kwargs['inc_cls_num'])
            new_fc.fc1.weight.data = self.network.classifier.weight.data
            new_fc.sigma.data = self.network.classifier.sigma.data
            self.network.classifier = new_fc
            lamda_mult = out_features*1.0 / self.kwargs['inc_cls_num']


        elif task_idx > 1:
            self.ref_model = copy.deepcopy(self.network) 
            in_features = self.network.classifier.in_features
            out_features1 = self.network.classifier.fc1.out_features
            out_features2 = self.network.classifier.fc2.out_features
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, self.kwargs['inc_cls_num']).to(self.device)
            new_fc.fc1.weight.data[:out_features1] = self.network.classifier.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.network.classifier.fc2.weight.data
            new_fc.sigma.data = self.network.classifier.sigma.data
            self.network.classifier = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (self.kwargs['inc_cls_num'])
        
        if task_idx > 0:
            self.cur_lamda = self.kwargs['lamda'] * math.sqrt(lamda_mult)
        else:
            self.cur_lamda = self.kwargs['lamda']

        self._init_new_fc(task_idx, buffer, train_loader)

        if task_idx == 0:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn1 = nn.CosineEmbeddingLoss()
            self.loss_fn2 = nn.CrossEntropyLoss()
            self.loss_fn3 = nn.MarginRankingLoss(margin=self.kwargs['dist'])

            self.ref_model.eval()
            self.num_old_classes = self.ref_model.classifier.out_features
            self.handle_ref_features = self.ref_model.classifier.register_forward_hook(get_ref_features)
            self.handle_cur_features = self.network.classifier.register_forward_hook(get_cur_features)
            self.handle_old_scores_bs = self.network.classifier.fc1.register_forward_hook(get_old_scores_before_scale)
            self.handle_new_scores_bs = self.network.classifier.fc2.register_forward_hook(get_new_scores_before_scale)

        self.network = self.network.to(self.device)
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)

    def _init_new_fc(self, task_idx, buffer, train_loader):
        if task_idx == 0:
            return
        old_embedding_norm = self.network.classifier.fc1.weight.data.norm(dim=1, keepdim=True)  
        average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor) 
        feature_model = self.network.backbone 
        num_features = self.network.classifier.in_features
        novel_embedding = torch.zeros((self.kwargs['inc_cls_num'], num_features))

        tmp_datasets = copy.deepcopy(train_loader.dataset)
        for cls_idx in range(self.network.classifier.fc1.out_features, self.network.classifier.fc1.out_features + self.network.classifier.fc2.out_features):
            cls_dataset = train_loader.dataset
            task_data, task_target = cls_dataset.images, cls_dataset.labels
            cls_indices = np.where(np.array(task_target) == cls_idx) # tuple
            cls_data, cls_target = np.array([task_data[i] for i in cls_indices[0]]), np.array([task_target[i] for i in cls_indices[0]])
            tmp_datasets.images = cls_data
            tmp_datasets.labels = cls_target
            tmp_loader = DataLoader(tmp_datasets, batch_size=128, shuffle=False, num_workers=2)
            num_samples = cls_data.shape[0]
            cls_features = self._compute_feature(feature_model, tmp_loader, num_samples, num_features)
            norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
            cls_embedding = torch.mean(norm_features, dim=0)
            novel_embedding[cls_idx-self.network.classifier.fc1.out_features] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
        
        self.network.to(self.device)
        self.network.classifier.fc2.weight.data = novel_embedding.to(self.device)

    def _compute_feature(self, feature_model, loader, num_samples, num_features):
        feature_model.eval()
        features = np.zeros([num_samples, num_features])
        start_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inputs, labels = batch['image'], batch['label']
                inputs = inputs.to(self.device)
                features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(feature_model.feature(inputs).cpu())
                start_idx = start_idx+inputs.shape[0]
        assert(start_idx==num_samples)
        return features


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.network(x)

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            ref_outputs = self.ref_model(x)
            loss = self.loss_fn1(cur_features, ref_features.detach(), \
                    torch.ones(x.size(0)).to(self.device)) * self.cur_lamda
            
            loss += self.loss_fn2(logit, y)
            
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==logit.size())
            gt_index = torch.zeros(outputs_bs.size()).to(self.device)
            gt_index = gt_index.scatter(1, y.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)
            max_novel_scores = outputs_bs[:, self.num_old_classes:].topk(self.K, dim=1)[0]
            hard_index = y.lt(self.num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)
            
            if  hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, self.K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss += self.loss_fn3(gt_scores.view(-1, 1), \
                        max_novel_scores.view(-1, 1), torch.ones(hard_num*self.K, 1).to(self.device)) * self.lw_mr

        pred = torch.argmax(logit, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.handle_ref_features.remove()
            self.handle_cur_features.remove()
            self.handle_old_scores_bs.remove()
            self.handle_new_scores_bs.remove()

    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logit = self.network(x)
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def get_parameters(self, config):
        if self.task_idx > 0:
            #fix the embedding of old classes
            ignored_params = list(map(id, self.network.classifier.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, \
                    self.network.parameters())
            tg_params =[{'params': base_params, 'lr': 0.1, 'weight_decay': 5e-4}, \
                        {'params': self.network.classifier.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        else:
            tg_params = self.network.parameters()
        
        return tg_params