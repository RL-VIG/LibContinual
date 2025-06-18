import torch
from torch import nn

class Finetune(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.device = kwargs['device']
        self.kwargs = kwargs
    
    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.classifier(self.backbone(x)['features'])    
        loss = self.loss_fn(logit, y)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def inference(self, data):
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
        pass

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
    
    def get_parameters(self, config):
        train_parameters = []
        train_parameters.append({"params": self.backbone.parameters()})
        train_parameters.append({"params": self.classifier.parameters()})
        return train_parameters

