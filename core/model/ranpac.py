'''
@article{zhou2023revisiting,
    author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
    title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
    journal = {arXiv preprint arXiv:2303.07338},
    year = {2023}
}

Code Reference:
https://github.com/RanPAC/RanPAC

Note:
* The accuracy of in-epoch test beside task 0 is low because the model classifier head is only being trained in after_task
* If you want to use transformation of dataset in original ranpac implementation, which result in better performance, set the attribute '_use_org_trfms' to be True
'''

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):

        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

        self.use_RP = False
        self.W_rand = None

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.sigma.data.fill_(1)

    def forward(self, input):

        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = F.relu(input @ self.W_rand)
            else:
                assert 0, 'should not reach here, for now'
                inn = input
            out = F.linear(inn, self.weight)

        out = self.sigma * out

        return {'logits': out}

class Network(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.device = device
        self.classifier = None

        self.feature_dim = kwargs['embd_dim']

    def update_classifer(self, num_classes):

        self._cur_task_id += 1
        del self.classifier
        self.classifier = CosineLinear(self.feature_dim, num_classes).to(self.device)

    def get_feature(self, x):

        features = self.backbone(x)
        return features

    def forward(self, x):

        logits = []
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits

class RanPAC(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._network = Network(backbone, device, **kwargs)

        self.device = device
        self.first_session_training = kwargs["first_session_training"]
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.total_cls_num = kwargs['total_cls_num']
        self.task_num = kwargs["task_num"]
        #self.use_RP = kwargs["use_RP"]
        self.M = kwargs['M']

        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._use_org_trfms = True # set to True if you want to use original implementation transforms of data
        self._skip_train = False # this flag is used to skip training
        self._skip_test = False # this flag is used to skip in-epoch test

        self._network.to(self.device)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx == 0:
            self._classes_seen_so_far = self.init_cls_num
        elif task_idx > 0:
            self._classes_seen_so_far += self.inc_cls_num
        
        self._network.update_classifer(self._classes_seen_so_far)

        if task_idx == 0 and self.first_session_training:
            self._skip_train = False
        else:
            self._skip_train = True
            print(f"Not training on task {task_idx}")

        self._skip_test = True

        if self._use_org_trfms:
            train_loader.dataset.trfms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            for loader in test_loaders:
                loader.dataset.trfms = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])

    def observe(self, data):

        if self._skip_train:
            # set required_grad be True so that it can call backward() but don't do anything
            return None, 0., torch.tensor(0.0, device = self.device, requires_grad = True)

        inputs, targets = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(inputs)['logits']
        loss = F.cross_entropy(logits, targets)

        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return preds, acc, loss

    def inference(self, data):

        if self._skip_test:
            return None, 0.

        inputs, targets = data['image'].to(self.device), data['label']
        logits = self._network(inputs)['logits']
        _, preds = torch.max(logits, dim=1)

        correct = preds.cpu().eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return logits, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._skip_test = False
        self._known_classes = self._classes_seen_so_far

        if task_idx == 0:
            
            # Initialize attribute for random projection classifier
            self.W_rand = torch.randn(self._network.classifier.in_features, self.M) 
            self.Q = torch.zeros(self.M, self.init_cls_num)
            self.G = torch.zeros(self.M, self.M)

        else:
            self.Q = torch.cat((self.Q, torch.zeros(self.M, self.inc_cls_num)), dim=1)

        self.update_rp_classifier(train_loader, test_loaders[0].dataset.trfms)

    @torch.no_grad()
    def update_rp_classifier(self, train_loader, test_trfms):

        self._network.eval()
        train_loader.dataset.trfms = test_trfms

        self._network.classifier.use_RP = True
        self._network.classifier.W_rand = self.W_rand.to(self.device) # feature_dim x M

        feature_list, label_list = [], []
        for batch in train_loader:
            x, y = batch['image'].to(self.device), batch['label']
            feature_list.append(self._network.get_feature(x).cpu())
            label_list.append(y)
        feature_list, label_list = torch.cat(feature_list, dim = 0), torch.cat(label_list, dim = 0)
        
        label_list = F.one_hot(label_list, self._classes_seen_so_far).to(torch.float32) 
        
        proj_feature_list = F.relu(feature_list @ self.W_rand)

        self.Q += proj_feature_list.T @ label_list
        self.G += proj_feature_list.T @ proj_feature_list
        
        ridges = 10.0**np.arange(-8,9)
        num_val_samples = int(proj_feature_list.shape[0] * 0.8)
        losses = []
        Q_val = proj_feature_list[:num_val_samples, :].T @ label_list[:num_val_samples, :]
        G_val = proj_feature_list[:num_val_samples, :].T @ proj_feature_list[:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(self.M), Q_val).T #better nmerical stability than .inv
            Y_train_pred = proj_feature_list[num_val_samples:, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, label_list[num_val_samples:, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print(f"Optimal lambda: {ridge}")

        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.M), self.Q).T #better nmerical stability than .inv
        self._network.classifier.weight.data = Wo[:self._network.classifier.weight.shape[0], :].to(self.device) # num_classes x M

    def get_parameters(self, config):
        return self._network.parameters()