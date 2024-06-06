"""
@inproceedings{guo2022online,
  title={Online continual learning through mutual information maximization},
  author={Guo, Yiduo and Liu, Bing and Zhao, Dongyan},
  booktitle={International Conference on Machine Learning},
  pages={8109--8126},
  year={2022},
  organization={PMLR}
}
https://proceedings.mlr.press/v162/guo22g.html

Code Reference:
https://github.com/gydpku/OCM/blob/main/test_cifar10.py

We referred to the original author's code implementation and performed structural refactoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from core.model.buffer.onlinebuffer import OnlineBuffer
import math
import numbers
import numpy as np
from torch.autograd import Function
import torch.distributed as dist
import diffdist.functional as distops
from torchvision import transforms

if torch.__version__ >= '1.4.0':
    kwargs = {'align_corners': False}
else:
    kwargs = {}


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def rot_inner_all(x):
    num = x.shape[0]
    R = x.repeat(4, 1, 1, 1)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num,3, 2, 16, 32)
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]
    s2 = a[1]
    s1_1 = torch.rot90(s1, 2, (2, 3))
    s2_2 = torch.rot90(s2, 2, (2, 3))
    R[num: 2 * num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0, 1, 3, 2)
    R[3 * num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0, 1, 3, 2)
    R[2 * num: 3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, 32, 32).permute(0, 1, 3, 2)
    return R


def Rotation(x, y):
    num = x.shape[0]
    X = rot_inner_all(x)
    y = y.repeat(16)
    for i in range(1, 16):
        y[i * num:(i + 1) * num]+=1000 * i
    return torch.cat((X, torch.rot90(X, 1, (2, 3)), torch.rot90(X, 2, (2, 3)), torch.rot90(X, 3, (2, 3))), dim=0), y





def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor

        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
    '''
    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)
    sim_matrix = torch.mm(outputs, outputs.t())  

    return sim_matrix


def Supervised_NT_xent_n(sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)

        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
    '''
    device = sim_matrix.device
    labels1 = labels.repeat(2)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix/(denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)
    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    loss1 = 2 * torch.sum(Mask1 * sim_matrix) / (2 * B)
    return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) + loss1


def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        
        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
    '''
    device = sim_matrix.device
    labels1 = labels.repeat(2)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk
    sim_matrix = torch.exp(sim_matrix / temperature)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = - torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)
    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    return torch.sum(Mask1 * sim_matrix) / (2 * B)





def Supervised_NT_xent_pre(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)

        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
    '''
    device = sim_matrix.device
    labels1 = labels#.repeat(2)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk  
    sim_matrix = torch.exp(sim_matrix / temperature) 
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
    labels1 = labels1.contiguous().view(-1, 1)
    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    return torch.sum(Mask1 * sim_matrix) / (2 * B)



#########################################################
#                                                       #
#                        Model                          #
#                                                       #
#########################################################



class OCM_Model(nn.Module):

    def __init__(self, backbone, feat_dim, num_class, device):
        '''
        A OCM model consists of a backbone, a classifier and a self-supervised head
        '''
    
        super(OCM_Model, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_class)
        self.head = nn.Linear(feat_dim, 128)  # for self-supervise
        self.device = device
        self.simclr_aug = self.get_simclr_aug()


    def get_features(self, x):
        out = self.backbone(x)['features']
        return out
    

    def forward_head(self, x):
        feat = self.get_features(x)
        out = self.head(feat)
        return feat, out


    def forward_classifier(self, x):
        feat = self.get_features(x)
        logits = self.classifier(feat)
        return logits
    

    def get_simclr_aug(self):
        hflip = transforms.RandomHorizontalFlip()
        color_gray = transforms.RandomGrayscale(p=0.25)
        # resize_crop = transforms.RandomResizedCrop(size=(32, 32), scale=(0.3, 1.0), antialias=True)
        resize_crop = transforms.RandomResizedCrop(size=(32, 32), scale=(0.3, 1.0))
        
        simclr_aug = torch.nn.Sequential(
            hflip,
            color_gray,
            resize_crop
        )
        return simclr_aug




class OCM(nn.Module):

    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super(OCM, self).__init__()
        
        # device setting
        self.device = kwargs['device']
        
        # current task index
        self.cur_task_id = 0

        # # current task class indexes
        # self.cur_cls_indexes = None
        
        # Build model structure
        self.model = OCM_Model(backbone, feat_dim, num_class, self.device)
        
        # Store old network
        self.previous_model = None

        # Store all seen classes
        self.class_holder = []

        self.buffer_per_class = 7


        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']
        self.task_num     = kwargs['task_num']
        
    
    

    def observe(self, data):
        # get data and labels
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        # update seen classes
        Y = deepcopy(y)
        for j in range(len(Y)):
            if Y[j] not in self.class_holder:
                self.class_holder.append(Y[j].detach())


        # learning
        x = x.requires_grad_()

        if self.cur_task_id == 0:
            pred, acc, loss = self.observe_first_task(x, y)
        else:
            pred, acc, loss = self.observe_incremental_tasks(x, y)

        # sample data to buffer
        self.buffer.add_reservoir(x=x.detach(), y=y.detach(), task=self.cur_task_id)

        return pred, acc, loss
    


    def observe_first_task(self, x, y):
        """
        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
        """
        images1, rot_sim_labels = Rotation(x, y)
        images_pair = torch.cat([images1, self.model.simclr_aug(images1)], dim=0)
        rot_sim_labels = rot_sim_labels.cuda()
        feature_map,outputs_aux = self.model.forward_head(images_pair)
        simclr = normalize(outputs_aux) 
        feature_map_out = normalize(feature_map[:images_pair.shape[0]])
        num1 = feature_map_out.shape[1] - simclr.shape[1]
        id1 = torch.randperm(num1)[0]
        size = simclr.shape[1]
        sim_matrix = torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())
        sim_matrix += get_similarity_matrix(simclr)
        loss_sim1 = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)
        lo1 = loss_sim1
        y_pred = self.model.forward_classifier(self.model.simclr_aug(x))
        loss = F.cross_entropy(y_pred, y) + lo1
        pred = torch.argmin(y_pred, dim=1)
        acc = torch.sum(pred == y).item() / x.size(0)
    
        return y_pred, acc, loss


    
    def observe_incremental_tasks(self, x, y):
        """
        Code Reference:
        https://github.com/gydpku/OCM/blob/main/test_cifar10.py
        """
        buffer_batch_size = min(64, self.buffer_per_class*len(self.class_holder))
        mem_x, mem_y,_ = self.buffer.sample(buffer_batch_size, exclude_task=None)
        mem_x = mem_x.requires_grad_()
        images1, rot_sim_labels = Rotation(x, y) 
        images1_r, rot_sim_labels_r = Rotation(mem_x,
                                               mem_y)
        images_pair = torch.cat([images1, self.model.simclr_aug(images1)], dim=0)
        images_pair_r = torch.cat([images1_r, self.model.simclr_aug(images1_r)], dim=0)
        t = torch.cat((images_pair,images_pair_r),dim=0)
        feature_map, u = self.model.forward_head(t)
        pre_u_feature, pre_u = self.previous_model.forward_head(images1_r)
        feature_map_out = normalize(feature_map[:images_pair.shape[0]])
        feature_map_out_r = normalize(feature_map[images_pair.shape[0]:])
        images_out = u[:images_pair.shape[0]]
        images_out_r = u[images_pair.shape[0]:]
        pre_u = normalize(pre_u)
        simclr = normalize(images_out)
        simclr_r = normalize(images_out_r)
        num1 = feature_map_out.shape[1] - simclr.shape[1]
        id1 = torch.randperm(num1)[0]
        id2 = torch.randperm(num1)[0]
        size = simclr.shape[1]

        sim_matrix = torch.matmul(simclr, feature_map_out[:, id1:id1 + size].t())
        sim_matrix_r = torch.matmul(simclr_r, feature_map_out_r[:, id2:id2 + size].t())
        sim_matrix += get_similarity_matrix(simclr)  
        sim_matrix_r +=  get_similarity_matrix(simclr_r)
        sim_matrix_r_pre = torch.matmul(simclr_r[:images1_r.shape[0]],pre_u.t())
        loss_sim_r =Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
        loss_sim_pre = Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
        loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)
        lo1 = loss_sim_r + loss_sim + loss_sim_pre
        y_label = self.model.forward_classifier(self.model.simclr_aug(mem_x))
        y_label_pre = self.previous_model.forward_classifier(self.model.simclr_aug(mem_x))
        loss =  F.cross_entropy(y_label, mem_y) + lo1 + F.mse_loss(y_label_pre[:, :self.prev_cls_num],
                                                                            y_label[:,
                                                                            :self.prev_cls_num])
        
        with torch.no_grad():
            logits = self.model.forward_classifier(x)[:, :self.accu_cls_num]
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == y).item() / x.size(0)
        return logits, acc, loss




    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model.forward_classifier(x)
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)
    

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        # load buffer to the models
        if self.cur_task_id == 0:
            self.buffer = buffer

        if self.cur_task_id == 0:
            self.accu_cls_num = self.init_cls_num
        else:
            self.accu_cls_num += self.inc_cls_num


    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        self.prev_cls_num = self.accu_cls_num
        self.cur_task_id += 1
        self.previous_model = deepcopy(self.model)


    def get_parameters(self, config):
        return self.model.parameters()