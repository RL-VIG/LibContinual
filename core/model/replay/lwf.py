import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .finetune import Finetune

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SplitCosineLinear(nn.Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out



cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs



class LWF(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.feat_dim = feat_dim
        self.classifier = None
        self.old_fc = None
        # self.classifier = CosineLinear(feat_dim, kwargs['init_cls_num'])
        # self.K = kwargs['K']
        # self.lw_mr = kwargs['lw_mr']
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.known_cls_num = 0
        self.total_cls_num = 0
        self.old_backbone = None
        
    def copy(self,nn):
        return copy.deepcopy(nn)
    def freeze(self,nn):
        for param in nn.parameters():
            param.requires_grad = False
        nn.eval()
        return nn
    
    def update_fc(self):
        fc = nn.Linear(self.feat_dim, self.total_cls_num).to(self.device)
        if self.classifier is not None:
            # del self.old_fc
            self.old_fc = self.freeze(self.copy(self.classifier))
            old_out = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            fc.weight.data[:old_out] = weight
            # print(fc.bias.data.shape,bias.shape)
            fc.bias.data[:old_out] = bias
            # for param in self.old_fc.parameters():
            #     param.requires_grad = False
        # del self.classifier
        self.classifier = fc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        self.known_cls_num = self.total_cls_num
        self.total_cls_num = self.init_cls_num + self.task_idx*self.inc_cls_num
        self.update_fc()
        self.loss_fn = nn.CrossEntropyLoss()
        if task_idx!=0:
            self.old_backbone = self.freeze(self.copy(self.backbone)).to(self.device)


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.shape)
        logit = self.classifier(self.backbone(x)['features'])    

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            fake_targets = y - self.known_cls_num
            loss_clf = self.loss_fn(logit[:,self.known_cls_num:],fake_targets)
            loss_kd = self._KD_loss(logit[:,:self.known_cls_num],self.old_fc(self.old_backbone(x)['features']),T=2)
            lamda = 3
            loss = lamda*loss_kd + loss_clf

        # print(logit)
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    def _cross_entropy(self, pre, logit):
        loss = None
        return loss