'''
Adapted from https://github.com/jxhe/unify-parameter-efficient-tuning

TODO: merge this with vit_adapter, either replace one with another
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.alexnet import Linear_TRGP

class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, 64)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output

'''
class MaskedAdapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, 64)
        self.scale_proj = nn.Linear(64, 64, bias=False)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

        self.identity_matrix = torch.eye(self.scale_proj.weight.shape[1])
        self.input_matrix = None
        self.space = []
        self.scale_param = nn.ParameterList()

    def enable_scale(self, space):
        self.space = space
        self.scale_param = nn.ParameterList([nn.Parameter(self.identity_matrix).to(self.scale_proj.weight.device) for _ in self.space])

    def disable_scale(self):
        self.space = []
        self.scale_param = nn.ParameterList()

    def forward(self, x, add_residual=True, residual=None, compute_input_matrix=False):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)

        if compute_input_matrix:
            self.input_matrix = down.clone().detach().cpu()

        scale_proj_weight = self.scale_proj.weight
        for scale, space in zip(self.scale_param, self.space):

            cropped_scale = scale[:space.shape[1], :space.shape[1]]
            cropped_identity_matrix = self.identity_matrix[:space.shape[1], :space.shape[1]].to(self.scale_proj.weight.device)

            scale_proj_weight = scale_proj_weight + scale_proj_weight @ space @ (cropped_scale - cropped_identity_matrix) @ space.T

        down = F.linear(down, scale_proj_weight)

        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output
'''

class MaskedAdapter(Adapter):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__(d_model, bottleneck, dropout, init_option, adapter_scalar, adapter_layernorm_option)

        self.down_proj = Linear_TRGP(self.n_embd, 64)
        self.up_proj = Linear_TRGP(self.down_size, self.n_embd)

    def forward(self, x, add_residual=True, residual=None, compute_input_matrix=False):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x, compute_input_matrix)
        down = self.non_linear_func(down)

        up = self.up_proj(down, compute_input_matrix)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output