'''
@misc{mcdonnell2024ranpacrandomprojectionspretrained,
      title={RanPAC: Random Projections and Pre-trained Models for Continual Learning}, 
      author={Mark D. McDonnell and Dong Gong and Amin Parveneh and Ehsan Abbasnejad and Anton van den Hengel},
      year={2024},
      eprint={2307.02251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2307.02251}, 
}

Code Reference:
https://github.com/RanPAC/RanPAC
'''

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.transformer import MultiHeadAttention_LoRA, VisionTransformer
from .backbone.clip import CLIP, tokenize
from .backbone.vit import ViTZoo, ViT_in21k_adapter

VIT = ViT_in21k_adapter
CLIP = CLIP

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):

        super().__init__()
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

        return out

class Network(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.device = device
        self.classifier = None

        if isinstance(self.backbone, VIT):
            self.feature_dim = self.backbone.feat_dim
        elif isinstance(self.backbone, CLIP):
            # Assuming the final features_dim is concat of image and text
            self.feature_dim = self.backbone.visual.output_dim + self.backbone.transformer.width
            self.accm_class_names = []   
            self.curr_class_names = []
            self.accm_text_tokens = None
            self.curr_text_tokens = None

            self.prompt_template = kwargs['prompt_template']

    def update_classifer(self, num_classes, train_loader):

        if isinstance(self.backbone, VIT):
            pass
        elif isinstance(self.backbone, CLIP):
            self.curr_class_names = train_loader.dataset.get_class_names()
            self.accm_class_names += self.curr_class_names

            self.curr_text_tokens = tokenize(
                [self.prompt_template.format(c) for c in self.curr_class_names]
            ).to(self.device)

            self.accm_text_tokens = tokenize(
                [self.prompt_template.format(c) for c in self.accm_class_names]
            ).to(self.device)
        else:
            assert 0

        self._cur_task_id += 1
        del self.classifier
        self.classifier = CosineLinear(self.feature_dim, num_classes).to(self.device)

    def get_feature(self, x):

        if isinstance(self.backbone, VIT):
            return self.backbone(x)
        elif isinstance(self.backbone, CLIP):
            features_image, features_text, logits_per_image, logits_per_text = self.backbone(x, self.curr_text_tokens)

            max_indices = logits_per_image.softmax(dim=-1).argmax(dim=1) # Shape will be [48]
            max_features = features_text[max_indices]  # Shape will be [48, 768]

            return torch.cat([features_image, max_features], dim=1)  # Shape will be [48, 1536]
        else:
            assert 0

    def forward(self, x, inference=False):

        if isinstance(self.backbone, VIT):
            features = self.backbone(x)
        elif isinstance(self.backbone, CLIP):
            if inference:
                features_image, features_text, logits_per_image, logits_per_text = self.backbone(x, self.accm_text_tokens)
            else:
                features_image, features_text, logits_per_image, logits_per_text = self.backbone(x, self.curr_text_tokens)

            max_indices = logits_per_image.softmax(dim=-1).argmax(dim=1) # Shape will be [48]
            max_features = features_text[max_indices]  # Shape will be [48, 768]
            features = torch.cat([features_image, max_features], dim=1)  # Shape will be [48, 1536]
        else:
            assert 0

        return self.classifier(features)

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
        self._skip_train = False # this flag is used to skip training

        self._network.to(self.device)

        if isinstance(backbone, CLIP):
            for name, param in self._network.named_parameters():
                if 'adapt' not in name:
                    param.requires_grad = False


    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx == 0:
            self._classes_seen_so_far = self.init_cls_num
        elif task_idx > 0:
            self._classes_seen_so_far += self.inc_cls_num
        
        self._network.update_classifer(self._classes_seen_so_far, train_loader)

        if task_idx == 0 and self.first_session_training:
            self._skip_train = False
        else:
            self._skip_train = True
            print(f"Not training on task {task_idx}")

    def observe(self, data):

        if self._skip_train:
            # set required_grad be True so that it can call backward() but don't do anything
            return None, 0., torch.tensor(0., device = self.device, requires_grad = True)

        inputs, targets = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(inputs)
        loss = F.cross_entropy(logits, targets)

        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return preds, acc, loss

    def inference(self, data):

        inputs, targets = data['image'].to(self.device), data['label']
        logits = self._network(inputs, True)
        _, preds = torch.max(logits, dim=1)

        correct = preds.cpu().eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return logits, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._known_classes = self._classes_seen_so_far

        if task_idx == 0:
            
            # Initialize attribute for random projection classifier
            self.W_rand = torch.randn(self._network.classifier.in_features, self.M) 
            self.Q = torch.zeros(self.M, self.init_cls_num) # C
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