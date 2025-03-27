"""
@inproceedings{liang2023adaptive,
  title={Adaptive Plasticity Improvement for Continual Learning},
  author={Liang, Yan-Shuo and Li, Wu-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7816--7825},
  year={2023}
}

Code Reference:
https://github.com/liangyanshuo/Adaptive-Plasticity-Improvement-for-Continual-Learning
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .backbone.alexnet import Conv2d_API, Linear_API, AlexNet_API

batch_list = [2*12, 100, 100] 
ksize = [4, 3, 2, 1, 1] # kernel size of each conv layer
channels = [3, 64, 128, 1024, 2048]
conv_output_size = [29, 12, 5] # output size of each conv layer

class Network(nn.Module):

    def __init__(self, backbone, **kwargs):

        super().__init__()
        self.backbone = backbone

        self.classifiers = nn.ModuleList([
            nn.Linear(backbone.feat_dim, kwargs['init_cls_num'], bias = False)] + 
            [nn.Linear(backbone.feat_dim, kwargs['inc_cls_num'], bias = False) for _ in range(kwargs['task_num'] - 1)]
        )

    def forward(self, data, t, compute_input_matrix = False):

        feat = self.backbone(data, t, compute_input_matrix)
        return [fc(feat) for fc in self.classifiers]

class API(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.network = Network(backbone, **kwargs)
        self.device = device

        self.task_num = kwargs["task_num"]
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self._known_classes = 0

        self.feature_list = []
        self.feature_mat = []
        self.project_type = []
        self.step = 0.5
        self.K = 10

        self.layers = [module for module in self.network.modules() if isinstance(module, Conv2d_API) or isinstance(module, Linear_API)]

        self.network.to(self.device)

    def observe(self, data, stage=0):

        # Stage=0 : The main train
        # Stage=1 : The FIRst train
        # Stage=2 : The Second train

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        if stage == 1 or stage == 2: # evaluate should only in stage==2
            logits = self.network(x, self.cur_task - 1)
        else:
            logits = self.network(x, self.cur_task)
            
        loss = F.cross_entropy(logits[self.cur_task], y)

        preds = logits[self.cur_task].max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        loss.backward()

        per_layer_norm = [layer.weight.grad.norm(p=2) for layer in self.layers]

        if self.cur_task > 0:
            for i, layer in enumerate(self.layers):
                sz =  layer.weight.grad.data.size(0)
                expand = self.expand[i][-1]
                assert expand == self.expand[i][self.cur_task-1]
                if self.project_type[i] == 'retain':
                    layer.weight.grad.data[:, :expand] = (layer.weight.grad.data[:,:expand].view(sz, -1) @ self.feature_mat[i]).view(layer.weight[:, :expand].size())
                elif self.project_type[i] == 'remove':                    
                    layer.weight.grad.data[:, :expand] = (layer.weight.grad.data[:,:expand].view(sz, -1) -
                                                         layer.weight.grad.data[:,:expand].view(sz, -1) @ self.feature_mat[i]).view(layer.weight[:, :expand].size())
                     
        for i, layer in enumerate(self.layers):
            self.per_layer_retain[i] += layer.weight.grad.norm(p=2)/per_layer_norm[i]

        if stage == 1:
            self.optimizer_stage1.step()
        else: 
            # either stage 0 or stage 2, stage 0 call optimizer.step() and stage 2 do nothing
            return preds, acc, loss

    def inference(self, data, task_id=-1):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        # Task-Aware (Task-Incremetanl Scenario)
        if task_id > -1:

            if task_id == 0:
                bias_classes = 0
            elif task_id == 1:
                bias_classes = self.init_cls_num
            else:
                bias_classes = self.init_cls_num + (task_id - 1) * self.inc_cls_num
            
            logits = self.network(x, task_id)
            preds = logits[task_id].max(1)[1] + bias_classes

        # Task-Agnostic (Class-Incremetanl Scenario)
        else:

            logits = torch.cat(self.network(x, self.cur_task), dim=-1)
            preds = logits.max(1)[1]
            
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self.per_layer_retain = [0., 0., 0., 0., 0.] # depends on backbone, if resnet then differerent
        self.cur_task = task_idx

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        if task_idx > 0:

            # bn's parameters are only learned for the first task
            for name, param in self.network.named_parameters():
                param.requires_grad_(True)
                if 'bn' in name:
                    param.requires_grad_(False)

            for ep in range(5):
                for batch in train_loader:
                    self.optimizer_stage1.zero_grad()
                    self.observe(batch, stage = 1)

                # TODO: early stop

            for batch in train_loader:
                self.observe(batch, stage = 2)
            
            num_iter = len(train_loader) * (5 + 1)
            self.per_layer_retain = [(retain/num_iter).item() for retain in self.per_layer_retain]

            mat_list = self.get_mat(task_idx - 1, train_loader)

            for i, mat in enumerate(mat_list):
                sz = mat.shape[-1]
                mat_list[i] = np.linalg.norm(
                    mat[:channels[i] * ksize[i] * ksize[i]].T.reshape(sz, channels[i], ksize[i], ksize[i]), ord=2, axis=(2,3)
                ).T

            sizes, ws = [], []
            for i, layer in enumerate(self.layers):

                U, _, _ = np.linalg.svd(mat_list[i], full_matrices=False)

                expand_dim = max((self.step - self.per_layer_retain[i]) * self.K, 0)
                size = max(min(math.ceil(expand_dim), channels[i]), 0)

                sizes.append(size)
                ws.append(torch.Tensor(U[:, :size]).to(self.device))

            self.network.backbone.expand(sizes, ws)
            self.network.to(self.device)

        self.layers = [module for module in self.network.modules() if isinstance(module, Conv2d_API) or isinstance(module, Linear_API)]

        # not include the additional w
        self.optimizer_stage1 = optim.SGD(self.get_parameters(additional=False), lr=0.01)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        mat_list = self.get_mat(task_idx, train_loader)

        self.expand = [] # self.expand[i][j] is the expanded size of i-th layer in j-th task
        for i, layer in enumerate(self.layers):
            self.expand.append(np.cumsum([0] + layer.expand))
            self.expand[i] += channels[i]

        for i, (feature, layer) in enumerate(zip(self.feature_list, self.layers)):
            assert task_idx > 0
            if isinstance(layer, Conv2d_API):
                sz = layer.expand[task_idx - 1] * ksize[i] * ksize[i]
            elif isinstance(layer, Linear_API):
                sz = layer.expand[task_idx - 1]
            else:
                raise NotImplementedError

            if sz:
                if self.project_type[i] == 'retain':
                    self.feature_list[i] = np.vstack((self.feature_list[i],np.zeros((sz, self.feature_list[i].shape[1]))))
                    self.feature_list[i] = np.hstack((self.feature_list[i],np.zeros((self.feature_list[i].shape[0], sz))))
                    self.feature_list[i][-sz:,-sz:] = np.eye(sz)
                elif self.project_type[i] == 'remove':
                    self.feature_list[i] = np.vstack((self.feature_list[i],np.zeros((sz,self.feature_list[i].shape[1]))))
                else:
                    raise Exception('Wrong project type')
            
        threshold = 0.97 + task_idx * 0.03 / self.task_num

        # get the space for each layer
        if task_idx == 0:
            for i, activation in enumerate(mat_list):

                U, S, _ = np.linalg.svd(activation, full_matrices = False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)

                if r < activation.shape[0]/2:
                    self.feature_list.append(U[:, :r])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:, r:])
                    self.project_type.append('retain')

        else:
            for i, activation in enumerate(mat_list):

                _, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()

                if self.project_type[i] == 'remove':

                    act_hat = activation - self.feature_list[i] @ self.feature_list[i].T @ activation
                    U, S, _ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total

                    if accumulated_sval >= threshold:
                        print (f'Skip Updating DualGPM for layer: {i+1}')
                    else:
                        r = np.sum(np.cumsum(sval_ratio) + accumulated_sval < threshold) + 1
                        Ui = np.hstack((self.feature_list[i], U[:, :r]))  
                        self.feature_list[i] = Ui[:, :min(Ui.shape[0], Ui.shape[1])]
     
                else:
                    act_hat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T) @ activation
                    U,S,_ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total     
                    accumulated_sval = sval_hat/sval_total          

                    if accumulated_sval < 1 - threshold:
                        print (f'Skip Updating Space for layer: {i+1}')
                    else:
                        r = np.sum(accumulated_sval - np.cumsum(sval_ratio) >= 1 - threshold) + 1
                        act_feature = self.feature_list[i] - U[:, :r] @ U[:, :r].T @ self.feature_list[i]
                        U, _, _ = np.linalg.svd(act_feature)
                        self.feature_list[i]=U[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                U, _, _ = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

        # Projection Matrix Precomputation
        self.feature_mat = []
        for feature, proj_type in zip(self.feature_list, self.project_type):
            if proj_type == 'remove':
                self.feature_mat.append(torch.Tensor(feature @ feature.T).to(self.device))
            elif proj_type == 'retain':
                self.feature_mat.append(torch.zeros(feature.shape[0], feature.shape[0]).to(self.device))

    def get_mat(self, t, train_loader):

        x = torch.cat([b['image'] for b in train_loader], dim = 0).to(self.device)

        # hardcoded, choose 125 input from it
        indices = torch.randperm(x.size(0))
        selected_indices = indices[:125]
        x = x[selected_indices]

        self.network.eval()
        self.network(x, t = t, compute_input_matrix = True)
        
        mat_list = [] # representation (activation) of each layer
        for i, module in enumerate(self.layers):
            
            if isinstance(module, Conv2d_API):
                bsz, ksz, s, inc = batch_list[i], ksize[i], conv_output_size[i], module.in_channels

                mat = np.zeros((ksz * ksz * inc, s * s * bsz))
                act = module.input_matrix.detach().cpu().numpy()

                k = 0
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk, :, ii:ksz+ii, jj:ksz+jj].reshape(-1) 
                            k += 1

                mat_list.append(mat)
            elif isinstance(module, Linear_API):
                mat_list.append(module.input_matrix.detach().cpu().numpy().T)

        return mat_list

    def get_parameters(self, config=None, additional=True):
        if additional:
            return self.network.parameters()
        else:
            return [param for name, param in self.network.named_parameters() if 'extra_ws' not in name]
        