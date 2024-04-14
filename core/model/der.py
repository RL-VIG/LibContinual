import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune
from core.model.backbone import resnet18, resnet34, resnet50
from core.utils import get_instance

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == "resnet18":
        dic = {"num_classes": 10, "args":{'dataset':'cifar100'}}
        return resnet18(**dic)
    # elif name=="resnet32":
    #     return resnet32()
    elif name == "resnet34":
        return resnet34()
    elif name == "resnet50":
        return resnet50()
    else:
        raise NotImplementedError("Unknown type {}".format(convnet_type))

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}
    
class DER(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.convnets = nn.ModuleList()
        self.pretrained = None
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []

        self.kwargs = kwargs
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.known_cls_num = 0
        self.total_cls_num = 0

        self.convnet_type = 'resnet18'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        logit = self.fc(features)['logits']    

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            loss_clf = self.loss_fn(logit, y)
            aux_targets = y.clone()
            aux_targets = torch.where(
                aux_targets - self.known_cls_num + 1 > 0,
                aux_targets - self.known_cls_num + 1,
                0,
            )
            aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]
            loss_aux = F.cross_entropy(aux_logits, aux_targets)
            loss = loss_aux + loss_clf

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss
    
    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        logit = self.fc(features)['logits']  
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.known_cls_num = self.total_cls_num
        self.total_cls_num = self.init_cls_num + self.task_idx*self.inc_cls_num

        self.freeze_conv()
        self.update_fc(self.total_cls_num)
        self.loss_fn = nn.CrossEntropyLoss()
        self.convnets = self.convnets.to(self.device)
        self.fc = self.fc.to(self.device)
        self.aux_fc = self.aux_fc.to(self.device)

    def _train(self):
        self.fc.train()
        # iffreeze('fc',self.fc)
        self.aux_fc.train()
        # iffreeze('auxfc',self.aux_fc)
        for i in range(self.task_idx -1):
            self.convnets[i].eval()
        self.convnets[-1].train()
        # for i,cov in enumerate(self.convnets):
        #     iffreeze(f'cov{i}',cov)
        
    def get_parameters(self, config):
        train_parameters = []
        
        train_parameters.append({"params": self.convnets.parameters()})
        
        if self.fc is not None:
            train_parameters.append({"params": self.fc.parameters()})
        if self.aux_fc is not None:
            train_parameters.append({"params": self.aux_fc.parameters()})
        return train_parameters
    
def iffreeze(name,net):
    for k,v in net.named_parameters():
        print('{}{}: {}'.format(name,k, v.requires_grad))