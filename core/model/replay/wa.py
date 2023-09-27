import torch
from torch import nn
import copy


class WA(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        # 用来初始化各自算法需要的对象
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = None
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.device = kwargs['device']
        self.task_idx = 0

    def observe(self, data):
        # 训练过程中，面对到来的一个batch的样本完成训练的损失计算以及参数更新，返回pred, acc, loss  预测结果，准确率，损失
        task_idx = self.task_idx
        self.classifier.to(self.device)
        if task_idx == 0:
            x, y = data['image'], data['label']
            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.classifier(self.backbone(x)['features'])
            loss = self.loss_fn(logit, y)
            pred = torch.argmax(logit, dim=1)

            acc = torch.sum(pred == y).item()
            return pred, acc / x.size(0), loss
        else:
            self.old_backbone.to(self.device)
            kd_lambda = self.known_classes / self.total_classes

            x, y = data['image'], data['label']
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.classifier(self.backbone(x)['features'])

            loss_ce = self.loss_fn(logits, y)
            loss_kd = KD_loss(logits[:, : self.known_classes], self.old_classifier(self.old_backbone(x)['features']))
            loss = (1 - kd_lambda) * loss_ce + kd_lambda * loss_kd

            pred = torch.argmax(logits, dim=1)

            acc = torch.sum(pred == y).item()
            return pred, acc / x.size(0), loss


    def inference(self, data):
        # 推理过程中，面对到来的一个batch的样本，完成预测，返回pred, acc
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        logit = self.classifier(self.backbone(x)['features'])
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.classifier(self.backbone(x)['features'])

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        # 可选，如果算法在每个任务开始前后有额外的操作，在这两个函数内完成
        self.total_classes = buffer.total_classes
        classifier = nn.Linear(self.feat_dim, self.total_classes)
        if self.classifier:
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            classifier.weight.data[: self.known_classes] = weight
            classifier.bias.data[: self.known_classes] = bias
        del self.classifier
        self.classifier = classifier

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        # 可选，如果算法在每个任务开始前后有额外的操作，在这两个函数内完成
        if task_idx > 0:
            increment = self.total_classes - self.known_classes
            weights = self.classifier.weight.data
            newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
            oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
            meannew = torch.mean(newnorm)
            meanold = torch.mean(oldnorm)
            gamma = meanold / meannew
            self.classifier.weight.data[-increment:, :] *= gamma

        # deepcopy
        copy_backbone = copy.deepcopy(self.backbone)
        copy_classifier = copy.deepcopy(self.classifier)
        # freeze
        for param in copy_backbone.parameters():
            param.requires_grad = False
        copy_backbone.eval()
        for param in copy_classifier.parameters():
            param.requires_grad = False
        copy_classifier.eval()
        self.old_classifier = copy_classifier
        self.old_backbone = copy_backbone
        self.known_classes = buffer.total_classes
        
        self.task_idx += 1

def KD_loss(pred, soft, T=2):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
