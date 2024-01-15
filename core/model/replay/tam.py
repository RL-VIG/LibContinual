import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune
from core.model.backbone.resnet import *
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

MINIBATCH_SIZE = 32

class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x

class AutoencoderSigmoid(BaseAttention):

    def __init__(self, input_dims=512, code_dims=256):
        super(AutoencoderSigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())
        


class TAM(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.reg_weight = self.kwargs['reg_weight']
        self.ema_update_freq = self.kwargs['ema_update_freq']
        self.ema_alpha = self.kwargs['ema_alpha']
        self.pairwise_weight = self.kwargs['pairwise_weight']
        self.beta = self.kwargs['beta']
        self.code_dims = self.kwargs['code_dims']
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.global_step = 0
        self.loss = F.cross_entropy
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, self.num_class)
        
        self.buffer = TAM_Buffer(500, self.device)
        for i in range(1 + (self.num_class-kwargs['init_cls_num'])//kwargs['inc_cls_num']):
            self.backbone.neck.append(AutoencoderSigmoid(input_dims=512, code_dims=self.code_dims))
            
        self.ema_model = copy.deepcopy(self.backbone).to(self.device)


    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        # self.backbone.neck.append(AutoencoderSigmoid(input_dims=512, code_dims=self.code_dims))
    
        self.task_idx = task_idx
        # in_features = self.backbone.fc.in_features
        # out_features = self.backbone.fc.out_features
        
        # new_fc = nn.Linear(in_features, self.kwargs['init_cls_num'] + task_idx * self.kwargs['inc_cls_num'])
        # new_fc.weight.data[:out_features] = self.backbone.fc.weight.data
        # self.backbone.fc = new_fc
        # self.backbone.to(self.device)

        
    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        feats = self.backbone.feature(x)
        # feats = self.backbone(x)['features']
        outputs_encoder = self.backbone.neck[self.task_idx].encoder(feats)
        outputs_neck = self.backbone.neck[self.task_idx].decoder(outputs_encoder)
        outputs = self.backbone.fc(outputs_neck * feats)
        loss = self.loss(outputs, y)

        # extra operations
        softmax = torch.nn.Softmax(dim=-1)
        for i in range(self.task_idx):
            outputs_i = self.backbone.neck[i](feats)
            pairwise_dist = torch.pairwise_distance(softmax(outputs_i.detach()),
                                                        softmax(outputs_neck), p=1).mean()
            loss -= self.pairwise_weight * (pairwise_dist)


        loss_1 = torch.tensor(0)
        if not self.buffer.is_empty():
            buf_outputs_1, buf_logits_1, _ = self.buffer_through_ae()
            loss_1 = self.reg_weight * F.mse_loss(buf_outputs_1, buf_logits_1.detach())
            loss += loss_1

            # CE for buffered images
            buf_outputs_2, _, buf_labels_2 = self.buffer_through_ae()
            loss += self.beta * self.loss(buf_outputs_2, buf_labels_2)


        task_labels = torch.ones(y.shape[0], device=self.device) * self.task_idx
        self.buffer.add_data(examples=x,
                             labels=y,
                             task_labels=task_labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()


        pred = torch.argmax(outputs, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss


    def buffer_through_ae(self):
        buf_inputs, buf_labels, task_labels = self.buffer.get_data(
            MINIBATCH_SIZE)
        buf_feats = self.backbone.feature(buf_inputs)
        buf_feats_ema = self.ema_model.feature(buf_inputs)
        buf_outout_ae = torch.zeros((MINIBATCH_SIZE, self.task_idx + 1, buf_feats.shape[-1]),
                                      device=self.device)
        buf_outout_ae_ema = torch.zeros((MINIBATCH_SIZE, self.task_idx + 1, buf_feats.shape[-1]),
                                      device=self.device)
        err_ae_1 = torch.zeros((MINIBATCH_SIZE, self.task_idx + 1), device=self.device)
        for i in range(self.task_idx + 1):
            out_ae_i = self.backbone.neck[i](buf_feats)
            out_ae_i_copy = self.ema_model.neck[i](buf_feats_ema.detach())
            recon_e = F.mse_loss(out_ae_i, buf_feats, reduction='none')
            err_ae_1[:, i] = torch.mean(recon_e, dim=1)
            buf_outout_ae[:, i, :] = out_ae_i
            buf_outout_ae_ema[:, i, :] = out_ae_i_copy

        # current model
        indices = torch.argmin(err_ae_1, dim=1)
        mask = F.one_hot(indices, self.task_idx + 1)
        mask = mask.unsqueeze(2).expand(-1, -1, buf_feats.shape[-1])
        buf_outout_ae = torch.sum(buf_outout_ae * mask, keepdim=True, dim=1).squeeze()
        buf_outputs = self.backbone.fc(buf_feats * buf_outout_ae)
        # EMA model
        mask_ema = F.one_hot(task_labels.long(), self.task_idx + 1)
        mask_ema = mask_ema.unsqueeze(2).expand(-1, -1, buf_feats_ema.shape[-1])
        buf_outout_ae_ema = torch.sum(buf_outout_ae_ema * mask_ema, keepdim=True, dim=1).squeeze()
        buf_logits = self.ema_model.fc(buf_feats_ema * buf_outout_ae_ema)

        return buf_outputs, buf_logits, buf_labels


    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.backbone.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass


    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        
        out_feats = self.backbone.feature(x)
        outout_ae = torch.zeros((out_feats.shape[0], self.task_idx+1, out_feats.shape[-1]),
                                          device=self.device)
        err_ae_1 = torch.zeros((out_feats.shape[0], self.task_idx+1), device=self.device)
        for i in range(self.task_idx+1):
            out_ae_i = self.backbone.neck[i](out_feats)
            recon_e = F.mse_loss(out_ae_i, out_feats, reduction='none')
            err_ae_1[:, i] = torch.mean(recon_e, dim=1)
            outout_ae[:, i, :] = out_ae_i
        
        indices = torch.argmin(err_ae_1, dim=1)
        mask1 = F.one_hot(indices, self.task_idx+1).unsqueeze(2)
        mask1 = mask1.expand(-1, -1, out_feats.shape[-1])
        outout_ae = torch.sum(outout_ae * mask1, keepdim=True, dim=1).squeeze()

        outputs = self.backbone.fc(out_feats * outout_ae)
        
        pred = torch.argmax(outputs, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)


    def _init_optim(self, config, task_idx):
        
        tg_params = self.backbone.parameters()
        return tg_params
    
    
    
    
    
    
    
class TAM_Buffer:
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples, labels,
                     logits, task_labels):
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size, transform=None):
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee)
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self):
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform):
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self):
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
        
        
        

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size