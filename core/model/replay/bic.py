import os
path = os.getcwd()
os.chdir(path)

import torch
from torch import nn
import logging
# from utils import logger
# from ...utils import logger
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ..backbone.resnet import BiasLayer
from ...data.dataset import BatchData, Exemplar
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader


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

class bic(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__()

        # device setting
        self.device = kwargs['device']
        
        # current task index
        self.cur_task_id = 0

        # current task class indexes
        self.cur_cls_indexes = None
        
        # Build model structure
        self.network = Model(backbone, feat_dim, num_class)
        
        # Store old network
        self.old_network = None
        
        # the previous class num before this task
        self.prev_cls_num = 0

        # the total class num containing this task
        self.accu_cls_num = 0

        # T
        self.T = 2

        # Load epoch
        self.epochs = kwargs['epoch']

        # Load optimizer parameters
        self.lr = kwargs['lr']
        self.gamma = kwargs['gamma']
        self.weight_decay = kwargs['weight_decay']
        self.momentum = kwargs['momentum']
        self.milestones = kwargs['milestones']
        
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']
        self.task_num     = kwargs['task_num']

        # class prototype vector
        self.class_means = None



    def observe(self, data):
        # get data and labels
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        # compute logits and loss
        logits, loss = self.criterion(x, y)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logits = self.network(x)
        pred = torch.argmax(logits, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)
 
    def forward(self, x):
        return self.network(x)[:, self.accu_cls_num]
    
    def before_task(self, task_idx, buffer, train_loader, test_loader):
        self._stage1_training(train_loader, test_loader)
        if self.cur_task_id == 0:
            self.accu_cls_num = self.init_cls_num
        else:
            self.accu_cls_num += self.inc_cls_num
            self._stage2_bias_correction(train_loader, test_loader)
        
        self.cur_cls_indexes = np.arange(self.prev_cls_num, self.accu_cls_num)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        # freeze old network as KD teacher
        self.old_network = deepcopy(self.network)
        self.old_network.eval()
        self.prev_cls_num = self.accu_cls_num
        
        # update buffer
        buffer.reduce_old_data(self.cur_task_id, self.accu_cls_num)
        val_transform = test_loaders[0].dataset.trfms
        buffer.update(self.network, train_loader, val_transform, 
                      self.cur_task_id, self.accu_cls_num, self.cur_cls_indexes,
                      self.device)
        
        # compute class mean vector via samples in buffer
        self.class_means = self.calc_class_mean(buffer,
                                               train_loader,
                                               val_transform,
                                               self.device).to(self.device)

        self.cur_task_id += 1
        self.lamda = self.prev_cls_num / self.accu_cls_num

    def criterion(self, x, y):
        # CE loss
        cur_logits = self.network(x)[:, :self.accu_cls_num]
        loss = F.cross_entropy(cur_logits, y)
        
        # For non-first tasks, using KD loss  
        if self.cur_task_id > 0:
            old_logits = self.old_network(x)
            old_target = F.sigmoid(old_logits)
            loss += F.binary_cross_entropy_with_logits(cur_logits[:, self.prev_cls_num],
                                                       old_target[:, self.prev_cls_num])
        
        return cur_logits, loss

    def _run(self, train_loader, test_loader, optimizer, scheduler, stage):
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            losses = 0.0
            for i, (image, label) in enumerate(train_loader):
                image, label = image.to(self.device), label.to(self.device)
                logits = self.network(image)

                if stage == "training":
                    clf_loss = F.cross_entropy(logits, label)
                    if self.old_network is not None:
                        old_logits = self.old_network(image).detach()

                        hat_pai_k = F.softmax(old_logits / self.T, dim=1)
                        log_pai_k = F.log_softmax(logits[:, : self.prev_cls_num] / self.T, dim=1)
                        distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))

                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss
                elif stage == "bias_correction":
                    loss = F.cross_entropy(torch.softmax(logits, dim=1), label)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            _, train_acc = self.inference(train_loader)
            _, test_acc = self.inference(test_loader)

            print("stage : {}, => Task {}, Epoch {}/{} => Loss {:.3f}, train_acc : {:.3f}, test_acc : {:.3f}".format(
                stage, self.cur_task_id, epoch, self.epochs, losses / len(train_loader), train_acc, test_acc))

    def _stage1_training(self, train_loader, test_loader):
        """
        if self.cur_task_id == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self.network.load_state_dict(loaded_dict['model_state_dict'])
            self.network.to(self.device)
            return
        """

        ignored_params = list(map(id, self.network.bias_layers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.network.parameters())
        network_params = [
            {"params": base_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {
                "params": self.network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },]
        optimizer = optim.SGD(network_params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.gamma)
        
        self.network.to(self.device)
        if self.old_network is not None:
            self.old_network.to(self.device)

        self._run(train_loader, test_loader, optimizer, scheduler, stage="training")

    def _stage2_bias_correction(self, train_loader, test_loader):
        if isinstance(self.network, nn.DataParallel):
            self.network = self.network.module
        network_params = [
            {
                "params": self.network.bias_layers[-1].parameters(),
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }]
        optimizer = optim.SGD(network_params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.gamma)
        
        self.network.to(self.device)
        self._run(train_loader, test_loader, optimizer, scheduler, stage="bias_correction" )