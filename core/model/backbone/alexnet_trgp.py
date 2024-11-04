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
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        # define the scale V
        size = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        self.identity_matrix = torch.eye(size, device = self.weight.device)

        self.space = []
        self.scale_param = nn.ParameterList()

        #self.scale1 = nn.Parameter(self.identity_matrix, requires_grad = True)
        #self.scale2 = nn.Parameter(self.identity_matrix, requires_grad = True)

    def enable_scale(self, space):
        self.space = space
        self.scale_param = nn.ParameterList([nn.Parameter(self.identity_matrix).to(self.weight.device) for _ in self.space])

    def disable_scale(self):
        self.space = []
        self.scale_param = nn.ParameterList()

    def forward(self, input, compute_input_matrix = False):
           
        # this should be only called once for each task
        if compute_input_matrix:
            self.input_matrix = input

        sz = self.weight.shape[0]

        masked_weight = self.weight

        for scale, space in zip(self.scale_param, self.space):

            cropped_scale = scale[:space.size(1), :space.size(1)]
            cropped_identity_matrix = self.identity_matrix[:space.shape[1], :space.shape[1]].to(self.weight.device)

            masked_weight = masked_weight + (self.weight.view(sz, -1) @ space @ (cropped_scale - cropped_identity_matrix) @ space.T).\
                                            view(self.weight.shape)

        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias = bias)

        # define the scale Q
        self.identity_matrix = torch.eye(self.weight.shape[1], device = self.weight.device)

        self.space = []
        self.scale_param = nn.ParameterList()

    def enable_scale(self, space):
        self.space = space
        self.scale_param = nn.ParameterList([nn.Parameter(self.identity_matrix).to(self.weight.device) for _ in self.space])

    def disable_scale(self):
        self.space = []
        self.scale_param = nn.ParameterList()

    def forward(self, input, compute_input_matrix = False):

        # this should be only called once for each task
        if compute_input_matrix:
            self.input_matrix = input # save input_matrix here
            
        masked_weight = self.weight

        for scale, space in zip(self.scale_param, self.space):

            cropped_scale = scale[:space.shape[1], :space.shape[1]]
            cropped_identity_matrix = self.identity_matrix[:space.shape[1], :space.shape[1]].to(self.weight.device)

            masked_weight = masked_weight + self.weight @ space @ (cropped_scale - cropped_identity_matrix) @ space.T

        return F.linear(input, masked_weight, self.bias)

class AlexNet_TRGP(nn.Module):

    def __init__(self, dropout_rate_1 = 0.2, dropout_rate_2 = 0.5):

        super().__init__()

        self.conv1 = Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, bias = False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats = False)
        
        self.conv2 = Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, bias = False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats = False)

        self.conv3 = Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2, bias = False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats = False)

        self.fc1 = Linear(in_features = 1024, out_features = 2048, bias = False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats = False)

        self.fc2 = Linear(in_features = 2048, out_features = 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats = False)

        # common use
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate_1)
        self.dropout2 = nn.Dropout(dropout_rate_2)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x, compute_input_matrix):

        x = self.conv1(x, compute_input_matrix)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.maxpool(x)
        
        x = self.conv2(x, compute_input_matrix)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.maxpool(x)

        x = self.conv3(x, compute_input_matrix)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x, compute_input_matrix)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x, compute_input_matrix)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout2(x)

        return x

