import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    
    def __init__(self,   
                in_channels, 
                out_channels,              
                kernel_size, 
                padding=0, 
                stride=1, 
                dilation=1,
                groups=1,                                                   
                bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels,
              kernel_size, stride=stride, padding=padding, bias=bias)
        # define the scale v
        size = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        scale = self.weight.data.new(size, size)
        scale.fill_(0.)
        # initialize the diagonal as 1
        scale.fill_diagonal_(1.)
        # self.scale1 = scale.cuda()
        self.scale1 = nn.Parameter(scale, requires_grad = True)
        self.scale2 = nn.Parameter(scale, requires_grad = True)
        self.noise = False
        if self.noise:
            self.alpha_w1 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)
            self.alpha_w2 = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.02, requires_grad = True)

    def forward(self, input, space1=None, space2=None):
           
        self.input_matrix = input # save input_matrix here

        if self.noise:
            with torch.no_grad():
                std = self.weight.std().item()
                noise = self.weight.clone().normal_(0,std)
        if space1 is not None or space2 is not None:
            sz =  self.weight.grad.data.size(0)

            if space2 is None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                #[chout, chinxkxk]  [chinxkxk, chinxkxk]
                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())
                if self.noise and self.training:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 

            if space1 is None:

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
     
                proj_weight = torch.mm(self.weight.view(sz,-1),norm_project).view(self.weight.size())
                diag_weight = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                if self.noise:
                    masked_weight = proj_weight + self.weight - diag_weight + self.alpha_w2 * noise * self.noise
                else:
                    masked_weight = proj_weight + self.weight - diag_weight 
            if space1 is not None and space2 is not None:
                real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
                norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
                proj_weight1 = torch.mm(self.weight.view(sz,-1),norm_project1).view(self.weight.size())
                diag_weight1 = torch.mm(self.weight.view(sz,-1),torch.mm(space1, space1.transpose(1,0))).view(self.weight.size())

                real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
                norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
                proj_weight2 = torch.mm(self.weight.view(sz,-1),norm_project2).view(self.weight.size())
                diag_weight2 = torch.mm(self.weight.view(sz,-1),torch.mm(space2, space2.transpose(1,0))).view(self.weight.size())

                if self.noise:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight + ((self.alpha_w2 + self.alpha_w1)/2) * noise * self.noise
                else:
                    masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight
       
        else:
            masked_weight = self.weight

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias = bias)

        # define the scale Q
        scale = self.weight.data.new(self.weight.size(1), self.weight.size(1))
        scale.fill_(0.)
        scale.fill_diagonal_(1.)

        self.scale1 = nn.Parameter(scale, requires_grad = True)
        self.scale2 = nn.Parameter(scale, requires_grad = True)

        self.input_matrix = torch.zeros(125 ,1024) # harcoded

    def forward(self, input, space1 = None, space2 = None, compute_input_matrix = False):

        self.input_matrix = input # save input_matrix here

        if input.dim() > 2:
            input = input.view(input.size(0), -1)

        # this should be only called once for each task
        if compute_input_matrix:
            self.input_matrix = input
            
        if space1 is not None and space2 is not None:
            real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
            norm_project1 = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))
            proj_weight1 = torch.mm(self.weight,norm_project1)
            diag_weight1 = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))
 
            real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
            norm_project2 = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
            proj_weight2 = torch.mm(self.weight,norm_project2)
            diag_weight2 = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

            masked_weight = proj_weight1 - diag_weight1 + proj_weight2 - diag_weight2 + self.weight

        elif space1 is not None:
            real_scale1 = self.scale1[:space1.size(1), :space1.size(1)]
            norm_project = torch.mm(torch.mm(space1, real_scale1), space1.transpose(1, 0))

            proj_weight = torch.mm(self.weight,norm_project)

            diag_weight = torch.mm(self.weight,torch.mm(space1, space1.transpose(1,0)))
            masked_weight = proj_weight + self.weight - diag_weight 

        elif space2 is not None:
            real_scale2 = self.scale2[:space2.size(1), :space2.size(1)]
            norm_project = torch.mm(torch.mm(space2, real_scale2), space2.transpose(1, 0))
    
            proj_weight = torch.mm(self.weight,norm_project)
            diag_weight = torch.mm(self.weight,torch.mm(space2, space2.transpose(1,0)))

            masked_weight = proj_weight + self.weight - diag_weight 

        else:
            masked_weight = self.weight
        
        return F.linear(input, masked_weight, self.bias)

class AlexNet_TRGP(nn.Module):

    def __init__(self, dropout_rate_1 = 0.2, dropout_rate_2 = 0.5):

        super().__init__()

        self.net = nn.ModuleList([
            Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, bias = False),
            nn.BatchNorm2d(64, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(dropout_rate_1),
            nn.MaxPool2d(kernel_size = 2),
            Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, bias = False),
            nn.BatchNorm2d(128, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(dropout_rate_1),
            nn.MaxPool2d(kernel_size = 2),
            Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2, bias = False),
            nn.BatchNorm2d(256, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(dropout_rate_2),
            nn.MaxPool2d(kernel_size = 2),
            Linear(in_features = 1024, out_features = 2048, bias = False),
            nn.BatchNorm1d(2048, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(dropout_rate_2),
            Linear(in_features = 2048, out_features = 2048, bias=False),
            nn.BatchNorm1d(2048, track_running_stats = False),
            nn.ReLU(),
            nn.Dropout(dropout_rate_2)]
        )

    def forward(self, x, space1, space2):

        for module in self.net:
            if isinstance(module, Conv2d) or isinstance(module, Linear):
                x = module(x, space1, space2)
            else:
                x = module(x)

        return x