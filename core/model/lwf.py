import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune

class LWF(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(self.feat_dim, kwargs['init_cls_num'])
        self.old_fc = None
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.known_cls_num = 0
        self.total_cls_num = 0
        self.old_backbone = None        

    def freeze(self,nn):
        for param in nn.parameters():
            param.requires_grad = False
        nn.eval()
        return nn
    
    def update_fc(self):
        fc = nn.Linear(self.feat_dim, self.total_cls_num).to(self.device)
        if self.classifier is not None:
            # del self.old_fc
            self.old_fc = self.freeze(copy.deepcopy(self.classifier))
            old_out = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            fc.weight.data[:old_out] = weight
            fc.bias.data[:old_out] = bias

        # del self.classifier
        self.classifier = fc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.known_cls_num = self.total_cls_num
        self.total_cls_num = self.init_cls_num + self.task_idx*self.inc_cls_num
        self.update_fc()
        self.loss_fn = nn.CrossEntropyLoss()
        if task_idx != 0:
            self.old_backbone = self.freeze(copy.deepcopy(self.backbone)).to(self.device)


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.classifier(self.backbone(x)['features'])    

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            fake_targets = y - self.known_cls_num
            loss_clf = self.loss_fn(logit[:,self.known_cls_num:],fake_targets)
            loss_kd = self._KD_loss(logit[:,:self.known_cls_num],self.old_fc(self.old_backbone(x)['features']),T=2)
            lamda = 3
            loss = lamda*loss_kd + loss_clf

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    def _cross_entropy(self, pre, logit):
        loss = None
        return loss
