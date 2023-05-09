import torch
from torch import nn

class Finetune(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__()
        # print("==============\n")
        # print(kwargs)
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.classifier = nn.Linear(feat_dim, num_class)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
    
    def observe(self, data):
        x, y = data['image'], data['label']
        # print(x.shape)
        logit = self.classifier(self.backbone(x)['features'])    
        loss = self.loss_fn(logit, y)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'], data['label']
        logit = self.classifier(self.backbone(x)['features'])  
        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.classifier(self.backbone(x)['features'])  