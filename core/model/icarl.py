"""
@inproceedings{rebuffi2017icarl,
  title={icarl: Incremental classifier and representation learning},
  author={Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander and Sperl, Georg and Lampert, Christoph H},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={2001--2010},
  year={2017}
}
https://arxiv.org/abs/1611.07725
"""

from typing import Iterator
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
import PIL
import os
import copy

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
    
    

class ICarl(nn.Module):
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

        
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']
        self.task_num     = kwargs['task_num']

        # class prototype vector
        self.class_means = None


    # only the current model is optimized
    def get_parameters(self, config):
        return self.network.parameters()
    
    
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
        
        # if self.class_means is not None:
        #     print(len(self.class_means), self.accu_cls_num)
        
        if self.class_means is not None and len(self.class_means) == self.accu_cls_num:
            # we only test when class mean vector computation is finished.
            return self.NCM_classify(data)
        
        else:
            # class mean vector for this task have not computed yet, 
            # call this function after func "after_task" called, 
            # and return value of this "inference" function is computed 
            # via model forward logits
            x, y = data['image'], data['label']
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.network(x)[:, :self.accu_cls_num]
            pred = torch.argmax(logits, dim=1)

            acc = torch.sum(pred == y).item()
            return pred, acc / x.size(0)
        
    

    def NCM_classify(self, data):

        def metric(x, y):
            """Calculate the pair-wise euclidean distance between input tensor `x` and `y`.
            Args:
                x (Tensor): to be calculated for distance, with shape (N, D)
                y (Tensor): to be calculated for distance, with shape (M, D), where D is embedding size.

            Returns:
                pair euclidean distance tensor with shape (N, M) 
                and dist[i][j] represent the distance between x[i] and y[j]
            """
            n = x.size(0)
            m = y.size(0)
            x = x.unsqueeze(1).expand(n, m, -1)
            y = y.unsqueeze(0).expand(n, m, -1)
            return torch.pow(x - y, 2).sum(2)  # (N, M)

        # using NCM
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        feats = feats = self.network.backbone(x)['features']
        feats = feats.view(feats.size(0), -1)
        distance = metric(feats, self.class_means)

        pred = torch.argmin(distance, dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0)


    def forward(self, x):
        return self.network(x)[:, self.accu_cls_num]
    
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.cur_task_id == 0:
            self.accu_cls_num = self.init_cls_num
        else:
            self.accu_cls_num += self.inc_cls_num
        
        self.cur_cls_indexes = np.arange(self.prev_cls_num, self.accu_cls_num)



    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        # freeze old network as KD teacher
        
        self.old_network = copy.deepcopy(self.network)
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
        
    

    

    def criterion(self, x, y):
        def _KD_loss(pred, soft, T=2):
            """
            Compute the knowledge distillation (KD) loss between the predicted logits and the soft target.
            Code Reference:
            KD loss function is borrowed from: https://github.com/G-U-N/PyCIL/blob/master/models/icarl.py
            """
            pred = torch.log_softmax(pred / T, dim=1)
            soft = torch.softmax(soft / T, dim=1)
            return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

        cur_logits = self.network(x)[:, :self.accu_cls_num]
        loss_clf = F.cross_entropy(cur_logits, y)

        if self.cur_task_id > 0:
            old_logits = self.old_network(x)
            loss_kd = _KD_loss(
                cur_logits[:, : self.prev_cls_num],
                old_logits[:, : self.prev_cls_num],
            )
            loss = loss_clf + loss_kd
        else:
            loss = loss_clf

        return cur_logits, loss




    def calc_class_mean(self, buffer, train_loader, val_transform, device):

        # mini dataset simulating all samples in the buffer
        class miniBufferDataset(Dataset):
            def __init__(self, root, mode, image_list, label_list, transforms):
                self.data_root = root
                self.mode = mode
                self.images = image_list
                self.labels = label_list
                self.transforms = transforms
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                label = self.labels[idx]
                image = PIL.Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
                image = self.transforms(image)
                return {"image": image, "label": label}

            def __len__(self):
                return len(self.labels)
        
        root_path = train_loader.dataset.data_root
        mode = train_loader.dataset.mode
        image_list = buffer.images
        label_list = buffer.labels
        ds = miniBufferDataset(root_path, mode, image_list, label_list, val_transform)

        icarl_loader = DataLoader(ds, 
                                  batch_size=train_loader.batch_size, 
                                  shuffle=False,
                                  num_workers=train_loader.num_workers, 
                                  pin_memory=train_loader.pin_memory)

        
        # compute features for all training samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            self.network.eval()
            for data in icarl_loader:
                images = data['image'].to(device)
                labels = data['label'].to(device)
                feats = self.network.backbone(images)['features']
                # normalize
                extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                extracted_targets.extend(labels)

        extracted_features = torch.cat(extracted_features).cpu()
        extracted_targets = torch.stack(extracted_targets).cpu()

        all_class_means = []
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # add the exemplars to the set and normalize
            cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
            
            all_class_means.append(cls_feats_mean)
        
        return torch.stack(all_class_means)
