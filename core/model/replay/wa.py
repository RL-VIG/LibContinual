import torch
from torch import nn
import copy
from torch.nn import functional as F
import math
import numpy as np
from .finetune import Finetune

def KD_loss(pred, soft, T=2):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        
    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        logits = self.classifier(self.backbone(x)['features'])
        return logits

    def update_classifier(self, nb_classes):
        classifier = nn.Linear(self.feat_dim, nb_classes)
        if self.classifier is not None:
            nb_output = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            classifier.weight.data[:nb_output] = weight
            classifier.bias.data[:nb_output] = bias

        del self.classifier
        self.classifier = classifier
        
    def weight_align(self, increment):
        weights = self.classifier.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.classifier.weight.data[-increment:, :] *= gamma
        
    def forward(self, x):
        return self.get_logits(x)
    
    def get_logits(self, x):
        logits = self.classifier(self.backbone(x)['features'])
        return logits
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def extract_vector(self, x):
        return self.backbone(x)["features"]




class WA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        # 用来初始化各自算法需要的对象
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        # self.backbone = backbone
        self.network = Model(self.backbone, feat_dim, kwargs['init_cls_num'])
        self.device = kwargs['device']
        self.old_network = None
        self.known_classes = 0
        self.total_classes = 0
        self.task_idx = 0
        # For buffer update
        self.total_classes_indexes = 0

    def observe(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        self.network.to(self.device)
        if self.old_network:
            self.old_network.to(self.device)

        logits = self.network(x)
        loss = F.cross_entropy(logits, y)

        if self.task_idx > 0:
            kd_lambda = self.known_classes / self.total_classes
            loss_kd = KD_loss(
                logits[:, : self.known_classes],
                self.old_network(x),
            )
            loss = (1 - kd_lambda) * loss + kd_lambda * loss_kd


        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self.network(x)
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.network(x)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.total_classes = buffer.total_classes
        self.network.update_classifier(self.total_classes)

        self.total_classes_indexes = np.arange(self.known_classes, self.total_classes)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.network.weight_align(self.total_classes - self.known_classes)
        # self.old_network = self.network.copy().freeze()
        self.old_network = copy.deepcopy(self.network).freeze()
        self.known_classes = self.total_classes

        # update buffer
        buffer.reduce_old_data(self.task_idx, self.total_classes)
        val_transform = test_loaders[0].dataset.trfms
        buffer.update(self.network, train_loader, val_transform,
                      self.task_idx, self.total_classes, self.total_classes_indexes,
                      self.device)

        self.task_idx += 1


