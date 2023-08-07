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



class LUCIR(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.classifier = CosineLinear(feat_dim, kwargs['init_cls_num'])
        self.K = kwargs['K']
        self.lw_mr = kwargs['lw_mr']
        
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx

        if task_idx == 1:
            self.ref_model = copy.deepcopy(self)
            lamda_mult = self.classifier.out_features / self.kwargs['inc_cls_num']
            new_fc = SplitCosineLinear(self.feat_dim, self.classifier.out_features, self.kwargs['inc_cls_num']).to(self.device)
            new_fc.fc1.weight.data = self.classifier.weight.data
            # new_fc.sigma.data = self.classifier.fc.sigma.data
            new_fc.sigma.data = self.classifier.sigma.data
            self.classifier = new_fc

        elif task_idx > 1:
            self.ref_model = copy.deepcopy(self) # 应该带上classifier
            in_features = self.classifier.in_features
            out_features1 = self.classifier.fc1.out_features
            out_features2 = self.classifier.fc2.out_features
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, self.kwargs['inc_cls_num']).to(self.device)
            new_fc.fc1.weight.data[:out_features1] = self.classifier.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.classifier.fc2.weight.data
            new_fc.sigma.data = self.classifier.sigma.data
            self.classifier = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (self.kwargs['inc_cls_num'])
        
        if task_idx > 1:
            self.cur_lamda = self.kwargs['lamda'] * math.sqrt(lamda_mult)

        if task_idx == 0:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn1 = nn.CosineEmbeddingLoss()
            self.loss_fn2 = nn.CrossEntropyLoss()
            self.loss_fn3 = nn.MarginRankingLoss(margin=self.kwargs['dist'])

            self.ref_model.eval()
            self.num_old_classes = self.ref_model.classifier.out_features
            self.handle_ref_features = self.ref_model.classifier.register_forward_hook(get_ref_features)
            self.handle_cur_features = self.classifier.register_forward_hook(get_cur_features)
            self.handle_old_scores_bs = self.classifier.fc1.register_forward_hook(get_old_scores_before_scale)
            self.handle_new_scores_bs = self.classifier.fc2.register_forward_hook(get_new_scores_before_scale)
            # num_old_classes = self.ref_model.fc.out_features
            # handle_ref_features = self.ref_model.fc.register_forward_hook(get_ref_features)
            # handle_cur_features = self.classifier.register_forward_hook(get_cur_features)
            # handle_old_scores_bs = self.classifier.fc1.register_forward_hook(get_old_scores_before_scale)
            # handle_new_scores_bs = self.classifier.fc2.register_forward_hook(get_new_scores_before_scale)
        # update optimizer  todo

    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.shape)
        logit = self.classifier(self.backbone(x)['features'])    

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            ref_outputs = self.ref_model(x)
            loss = self.loss_fn1(cur_features, ref_features.detach(), \
                    torch.ones(x.size(0)).to(self.device))
            
            loss += self.loss_fn2(logit, y)
            
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==logit.size())
            gt_index = torch.zeros(outputs_bs.size()).to(self.device)
            gt_index = gt_index.scatter(1, y.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)
            max_novel_scores = outputs_bs[:, self.num_old_classes:].topk(self.K, dim=1)[0]
            hard_index = y.lt(self.num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)
            
            if  hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, self.K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss += self.loss_fn3(gt_scores.view(-1, 1), \
                max_novel_scores.view(-1, 1), torch.ones(hard_num*self.K, 1).to(self.device)) * self.lw_mr

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.handle_ref_features.remove()
            self.handle_cur_features.remove()
            self.handle_old_scores_bs.remove()
            self.handle_new_scores_bs.remove()