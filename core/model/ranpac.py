import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# the acc of in-epoch test beside task 0 is low bcs the model classifier head never being train, 
# it is only train in after_task, so only last test acc is high

# changing trfm in before_task only change the trfm in before_task, not in traning steps
# and theres no way to change the trfm in training step other than directly chaning in data.py 

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
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
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
        self.use_RP = kwargs["use_RP"]
        self.M = kwargs['M']

        self._known_classes = 0
        self._classes_seen_so_far = 0
        self._train_flag = False # flag to indicate if the model should do training
        self._use_trfm_org = False # use transformation of data in originial code

        '''
        If you want to use transformation of dataset in original ranpac implementation, which result in better performance, 
        replace the 'vit_train_transform' and 'vit_test_transform' in core/data/data.py with code below.

        vit_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        vit_test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        and set the 'self._use_trfm_org' be True
        '''

        self._network.to(self.device)

    def freeze_backbone(self):

        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = False

    def setup_RP(self):

        self._network.classifier.use_RP=True

        self._network.classifier.weight = nn.Parameter(torch.Tensor(self._network.classifier.out_features, self.M).to(self.device)) #num classes in task x M
        self._network.classifier.reset_parameters()

        self._network.classifier.W_rand=torch.randn(self._network.classifier.in_features, self.M).to(self.device)
        self.W_rand = copy.deepcopy(self._network.classifier.W_rand)
        self.Q = torch.zeros(self.M, self.total_cls_num)
        self.G = torch.zeros(self.M, self.M)
    
    def update_classifier(self, train_loader):

        self._network.eval() 

        self._network.classifier.use_RP=True
        self._network.classifier.W_rand = self.W_rand

        feature_list, label_list = [], []
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch['image'].to(self.device), batch['label']
            feature_list.append(self._network.get_feature(x).cpu())
            label_list.append(y)
            
        feature_list, label_list = torch.cat(feature_list, dim=0), torch.cat(label_list, dim=0)

        def target2onehot(targets, num_classes):
            onehot = torch.zeros(targets.shape[0], num_classes).to(targets.device)
            onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
            return onehot
        
        label_list = target2onehot(label_list, self.total_cls_num)
        proj_feature_list = torch.nn.functional.relu(feature_list @ self._network.classifier.W_rand.cpu())

        self.Q += proj_feature_list.T @ label_list
        self.G += proj_feature_list.T @ proj_feature_list

        def optimise_ridge_parameter(features, labels):
            ridges = 10.0**np.arange(-8,9)
            num_val_samples = int(features.shape[0]*0.8)
            losses = []
            Q_val = features[:num_val_samples,:].T @ labels[:num_val_samples,:]
            G_val = features[:num_val_samples,:].T @ features[:num_val_samples,:]
            for ridge in ridges:
                Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
                Y_train_pred = features[num_val_samples::,:] @ Wo.T
                losses.append(F.mse_loss(Y_train_pred, labels[num_val_samples::,:]))
            ridge = ridges[np.argmin(np.array(losses))]
            print(f"Optimal lambda: {ridge}")
            return ridge

        ridge = optimise_ridge_parameter(proj_feature_list, label_list)
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T #better nmerical stability than .inv
        self._network.classifier.weight.data = Wo[0:self._network.classifier.weight.shape[0],:].to(self.device)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx == 0:
            self._classes_seen_so_far = self.init_cls_num
        elif task_idx > 0:
            self._classes_seen_so_far += self.inc_cls_num
        
        self._network.update_classifer(self._classes_seen_so_far)

        if task_idx == 0 and self.first_session_training:
            self._train_flag = True

        if task_idx == 1 and self.first_session_training:
            #for name, param in self._network.named_parameters():
            #    param.requires_grad = False
            self._train_flag = False

        if not self._train_flag:
            print(f"Not training on task {task_idx}")

    def observe(self, data):

        if not self._train_flag:
            # set required_grad be True so that it can call backward() but don't do anything
            empty_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return None, 0., empty_loss

        self._network.train()

        # Mask learned classes, leaving only unlearned classes
        inputs, targets = data['image'].to(self.device), data['label'].to(self.device)
        mask = (targets >= self._known_classes).nonzero().view(-1)
        inputs = torch.index_select(inputs, 0, mask)
        targets = torch.index_select(targets, 0, mask) - self._known_classes

        logits = self._network(inputs)['logits']
        loss = F.cross_entropy(logits, targets)

        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return preds, acc, loss

    def inference(self, data):

        inputs, targets = data['image'].to(self.device), data['label']
        logits = self._network(inputs)['logits']
        _, preds = torch.max(logits, dim=1)

        correct = preds.cpu().eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return logits, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._known_classes = self._classes_seen_so_far

        if task_idx == 0:
            self.freeze_backbone()
            self.setup_RP()

        # Changing the transformation of data in train_loader to transformation of test data
        if self._use_trfm_org:
            train_loader.trfms = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor()])
        else:
            train_loader.trfms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0., 0., 0.), (1., 1., 1.))])

        # train_loader is now train_loader_for_CPs
        self.update_classifier(train_loader)

    def get_parameters(self, config):
        return self._network.parameters()