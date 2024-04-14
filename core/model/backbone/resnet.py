'''
Reference:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import math
import torch
import torch.nn as nn
import copy
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url

from torch.nn.parameter import Parameter
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,args=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        assert args is not None, "you should pass args to resnet"
        if 'cifar' in args["dataset"]:
            self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        elif 'imagenet' in args["dataset"]:
            if args["init_cls"] == args["increment"]:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # Removed in _forward_impl

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
                    
        self.neck = nn.ModuleList()
        self.fc = nn.Linear(512, 20)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)  # [bs, 64, 32, 32]

        x_1 = self.layer1(x)  # [bs, 128, 32, 32]
        x_2 = self.layer2(x_1)  # [bs, 256, 16, 16]
        x_3 = self.layer3(x_2)  # [bs, 512, 8, 8]
        x_4 = self.layer4(x_3)  # [bs, 512, 4, 4]

        pooled = self.avgpool(x_4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]
        # x = self.fc(x)

        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }

    def forward(self, x):
        return self._forward_impl(x)
    
    def feature(self, x):
        x = self.conv1(x)  # [bs, 64, 32, 32]

        x_1 = self.layer1(x)  # [bs, 128, 32, 32]
        x_2 = self.layer2(x_1)  # [bs, 256, 16, 16]
        x_3 = self.layer3(x_2)  # [bs, 512, 8, 8]
        x_4 = self.layer4(x_3)  # [bs, 512, 4, 4]

        pooled = self.avgpool(x_4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]
        
        return features

    @property
    def last_conv(self):
        if hasattr(self.layer4[-1], 'conv3'):
            return self.layer4[-1].conv3
        else:
            return self.layer4[-1].conv2


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    if 'cosine_fc' in kwargs['args'].keys() and kwargs['args']['cosine_fc']:
        in_features = model.fc.in_features
        out_features = model.fc.out_features
        model.fc = CosineLinear(in_features, out_features)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)




class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3):
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        # self.fc = nn.Linear(64*block.expansion, 20)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
                
                
        self.neck = nn.ModuleList()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            # downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        return {
            'fmaps': [x_1, x_2, x_3],
            'features': features
        }
        # out = self.fc(features)
        # return out
    
    
    def feature(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]
        
        return features

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())


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





# modifiled resnet source code from: 
# https://github.com/hshustc/CVPR19_Incremental_Learning

class modified_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(modified_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last: #remove ReLU in the last layer
            out = self.relu(out)

        return out

class modified_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(modified_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = modified_linear.CosineLinear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else: 
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return {"features": x}

        
    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x


def cifar_resnet20(pretrained=False, **kwargs):
    n = 3
    model = CifarResNet(ResNetBasicblock, 20)
    return model

def cifar_resnet32(pretrained=False, **kwargs):
    # EWC
    model = CifarResNet(ResNetBasicblock, 32)
    return model


def resnet32_V2(pretrained=False, **kwargs):
    # ## LUCIR:
    n = 5
    model = modified_ResNet(modified_BasicBlock, [n, n, n], num_classes=50)
    return model






#########################################################
# For Wa

def resnet32_wa(pretrained=False, **kwargs):
    return IncrementalNet()


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.convnet = resnet32()
        self.classifier = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.classifier(x["features"])
        out.update(x)

        return out

    def update_classifier(self, nb_classes):
        pass

    def generate_classifier(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet(BaseNet):
    def __init__(self):
        super().__init__()

    def update_classifier(self, nb_classes):
        classifier = self.generate_classifier(self.feature_dim, nb_classes)
        if self.classifier is not None:
            nb_output = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            classifier.weight.data[:nb_output] = weight
            classifier.bias.data[:nb_output] = bias

        del self.classifier
        self.classifier = classifier

    def weight_align(self, increment):
        weights = self.classifier.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.classifier.weight.data[-increment:, :] *= gamma

    def generate_classifier(self, in_dim, out_dim):
        classifier = SimpleLinear(in_dim, out_dim)

        return classifier

    def forward(self, x):
        x = self.convnet(x)
        out = self.classifier(x["features"])
        out.update(x)
        return out


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