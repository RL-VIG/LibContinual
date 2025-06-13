"""
@inproceedings{
    saha2021gradient,
    title={Gradient Projection Memory for Continual Learning},
    author={Gobinda Saha and Isha Garg and Kaushik Roy},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=3AOj0RCNC2}
}

Code Reference:
https://github.com/sahagobinda/GPM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone.alexnet import Conv2d_TRGP, Linear_TRGP

class Network(nn.Module):

    def __init__(self, backbone, **kwargs):

        super().__init__()
        self.backbone = backbone

        self.classifiers = nn.ModuleList([
            nn.Linear(backbone.feat_dim, kwargs['init_cls_num'], bias = False)] + 
            [nn.Linear(backbone.feat_dim, kwargs['inc_cls_num'], bias = False) for _ in range(kwargs['task_num'] - 1)]
        )

    def forward(self, data, compute_input_matrix = False):

        logits = []
        image_features = self.backbone(data, compute_input_matrix)
        for classifier in self.classifiers:
            logits.append(classifier(image_features))

        return logits

class GPM(nn.Module):

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

        self.layers = [] # 3 Conv2d, Then 2 Linear
        for module in self.network.modules():
            if isinstance(module, Conv2d_TRGP) or isinstance(module, Linear_TRGP):
                self.layers.append(module)

        self.network.to(self.device)

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self.network(x)
        loss = F.cross_entropy(logits[self.cur_task], y)

        preds = logits[self.cur_task].max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        loss.backward()
        
        if self.cur_task > 0:
            for i, module in enumerate(self.layers):
                sz = module.weight.grad.data.shape[0]
                module.weight.grad.data = module.weight.grad.data - (module.weight.grad.data.view(sz,-1) @ self.feature_mat[i]).view(module.weight.shape)

        return preds, acc, loss
    
    def inference(self, data, task_id = -1):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        # Task-Aware (Task-Incremetanl Scenario)
        if task_id > -1:

            if task_id == 0:
                bias_classes = 0
            elif task_id == 1:
                bias_classes = self.init_cls_num
            else:
                bias_classes = self.init_cls_num + (task_id - 1) * self.inc_cls_num
            
            logits = self.network(x)
            preds = logits[task_id].max(1)[1] + bias_classes

        # Task-Agnostic (Class-Incremetanl Scenario)
        else:

            logits = torch.cat(self.network(x), dim=-1)
            preds = logits.max(1)[1]
            
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self.cur_task = task_idx

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        if task_idx > 0:

            self.feature_mat = [torch.tensor(feat @ feat.T, dtype=torch.float32, device=self.device) for feat in self.feature_list] 
            
            for name, param in self.network.named_parameters():
                param.requires_grad_(True)
                if 'bn' in name:
                    param.requires_grad_(False)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        x = []
        for batch in train_loader:
            x.append(batch['image'].to(self.device))

        x = torch.cat(x, dim = 0)

        # hardcoded, choose 125 input from it
        indices = torch.randperm(x.size(0))
        selected_indices = indices[:125]
        x = x[selected_indices]

        self.network.eval()
        self.network(x, compute_input_matrix = True)

        batch_list = [2*12,100,100] 
        ksize = [4, 3, 2] # kernel size of each conv layer
        conv_output_size = [29, 12, 5] # output size of each conv layer
        in_channel = [3, 64, 128] # input channel of each conv layer

        mat_list = [] # representation (activation) of each layer

        for i, module in enumerate(self.layers):
            
            if isinstance(module, Conv2d_TRGP):
                bsz, ksz, s, inc = batch_list[i], ksize[i], conv_output_size[i], in_channel[i]

                # act is the input of each layer (both conv and linear)
                mat = np.zeros((ksz * ksz * inc, s * s * bsz))
                act = module.input_matrix.detach().cpu().numpy()

                k = 0
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk, :, ii:ksz+ii, jj:ksz+jj].reshape(-1) 
                            k += 1

                mat_list.append(mat)
            elif isinstance(module, Linear_TRGP):
                mat_list.append(module.input_matrix.detach().cpu().numpy().T)

        threshold = 0.97 + task_idx * 0.003

        # get the space for each layer
        if task_idx == 0:
            for i, activation in enumerate(mat_list):

                U, S, _ = np.linalg.svd(activation, full_matrices = False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)

                self.feature_list.append(U[:, :r])
        else:
            for i, activation in enumerate(mat_list):

                _, S, _ = np.linalg.svd(activation, full_matrices = False)
                sval_total = (S**2).sum()
                
                act_hat = activation - self.feature_list[i] @ self.feature_list[i].T @ activation
                U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total

                if accumulated_sval >= threshold:
                    print (f'Skip Updating GPM for layer: {i+1}')
                else:
                    r = np.sum(np.cumsum(sval_ratio) + accumulated_sval < threshold) + 1
                    Ui = np.hstack((self.feature_list[i], U[:, :r]))  
                    self.feature_list[i] = Ui[:, :min(Ui.shape[0], Ui.shape[1])]

    def get_parameters(self, config):
        return self.network.parameters()