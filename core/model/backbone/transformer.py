'''
Code Reference:

* https://github.com/jadore801120/attention-is-all-you-need-pytorch/
* https://github.com/GT-RIPL/CODA-Prompt
* https://github.com/openai/CLIP
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import Counter
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_, DropPath

from .petl.adapter import Adapter, MaskedAdapter
from .petl.proj import Proj

# Helper
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # 加权

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

# Sub-module of Attention
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, attn_mask=None, register_hook=False, prompt=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, NH, N, HD]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask.unsqueeze(0) # For head axis broadcasting

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadAttention_LoRA(MultiHeadAttention):

    '''
    Attention module with lora, apply to k, v
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora_rank=10, lora_bias=False):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

        self.lora_rank = lora_rank
        
        self.lora_A_k = nn.Linear(self.dim, self.lora_rank, bias=lora_bias)
        self.lora_B_k = nn.Linear(self.lora_rank, self.dim, bias=lora_bias)
        self.lora_A_v = nn.Linear(self.dim, self.lora_rank, bias=lora_bias)
        self.lora_B_v = nn.Linear(self.lora_rank, self.dim, bias=lora_bias)        
        self.apply_lora = False

        self.cur_matrix = torch.zeros(self.dim ,self.dim)
        self.n_cur_matrix = 0

    def init_param(self):

        nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k.weight)
        nn.init.zeros_(self.lora_B_v.weight)

        self.apply_lora = True

    def merge_weight(self):
        q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)
        k_weight = k_weight + self.lora_B_k.weight @ self.lora_A_k.weight
        v_weight = v_weight + self.lora_B_v.weight @ self.lora_A_v.weight
        self.qkv.weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)
        self.apply_lora = False

    def reset_input_matrix(self):
        self.cur_matrix.zero_()
        self.n_cur_matrix = 0

    def forward(self, x, attn_mask=None, register_hook=False, prompt=None, get_input_matrix = False):
    
        if get_input_matrix:
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]
            
        B, N, C = x.shape

        q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)

        if self.apply_lora:
            k_weight = k_weight + self.lora_B_k.weight @ self.lora_A_k.weight
            v_weight = v_weight + self.lora_B_v.weight @ self.lora_A_v.weight
        
        qkv = F.linear(x, torch.cat([q_weight, k_weight, v_weight], dim=0), self.qkv.bias.data).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask.unsqueeze(0) # For head axis broadcasting

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadAttention_MaskedLoRA(MultiHeadAttention_LoRA):

    '''
    Attention module with masked (projection) lora, apply to k, v
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora_rank=10, lora_bias=False):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, lora_rank, lora_bias)

        # Trgp implementation
        self.identity_matrix = torch.eye(self.qkv.weight.shape[1])

        self.space = [[torch.tensor((1)), torch.tensor((1))] for _ in range(10)]
        self.scale_param = nn.ModuleList([nn.ParameterList([nn.Parameter(self.identity_matrix) for _ in range(2)]) for _ in range(10)])
        self.scaling_mask = [[False, False] for _ in range(10)]

    def enable_scale(self, task_id, space):
        if len(space) == 2:
            self.space[task_id][0] = space[0]
            self.space[task_id][1] = space[1]
            self.scaling_mask[task_id][0] = True
            self.scaling_mask[task_id][1] = True
        elif len(space) == 1:
            self.space[task_id][0] = space[0]
            self.scaling_mask[task_id][0] = True

        for scale_param_list in self.scale_param:
            for scale_param in scale_param_list:
                scale_param = scale_param.to(self.qkv.weight.device)

    def forward(self, x, attn_mask=None, expert_id=0, register_hook=False, prompt=None, get_input_matrix = False):

        if get_input_matrix:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])            
            self.n_cur_matrix += x.shape[0]*x.shape[1]
            
        B, N, C = x.shape

        q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)

        if self.apply_lora:
            k_weight = k_weight + self.lora_B_k.weight @ self.lora_A_k.weight
            v_weight = v_weight + self.lora_B_v.weight @ self.lora_A_v.weight
        
        for mask, scale, space in zip(self.scaling_mask[expert_id], self.scale_param[expert_id], self.space[expert_id]):

            if not mask:
                break

            scale_size = space.shape[1]
            cropped_scale = scale[:scale_size, :scale_size]

            cropped_scale = cropped_scale @ cropped_scale.T # better, idk why

            cropped_identity_matrix = self.identity_matrix[:scale_size, :scale_size].to(self.qkv.weight.device)

            k_weight = k_weight + k_weight @ space @ (cropped_scale - cropped_identity_matrix) @ space.T
            v_weight = v_weight + v_weight @ space @ (cropped_scale - cropped_identity_matrix) @ space.T

        qkv = F.linear(x, torch.cat([q_weight, k_weight, v_weight], dim=0), self.qkv.bias.data).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask.unsqueeze(0) # For head axis broadcasting

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadAttention_MoEMaskedLoRA(MultiHeadAttention_LoRA):

    '''
    Attention module with masked (projection) lora, apply to k, v
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora_rank=10, lora_bias=False):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, lora_rank, lora_bias)

        # Trgp implementation
        self.identity_matrix = torch.eye(self.qkv.weight.shape[1])

        self.space = [[torch.tensor((1)), torch.tensor((1))] for _ in range(10)]
        self.scale_param = nn.ModuleList([nn.ParameterList([nn.Parameter(self.identity_matrix) for _ in range(2)]) for _ in range(10)])
        self.scaling_mask = [[False, False] for _ in range(10)]

        self.experts_num = 10
        self.top_k = 1
        self.router = nn.Parameter(torch.zeros(768, self.experts_num), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(768, self.experts_num), requires_grad=True)
        self.experts = nn.ModuleList([
            Proj(d_model=768)
            for _ in range(self.experts_num)                           
        ])
        self.softmax = nn.Softmax(1)

    def enable_scale(self, task_id, space):
        if len(space) == 2:
            self.space[task_id][0] = space[0]
            self.space[task_id][1] = space[1]
            self.scaling_mask[task_id][0] = True
            self.scaling_mask[task_id][1] = True
        elif len(space) == 1:
            self.space[task_id][0] = space[0]
            self.scaling_mask[task_id][0] = True

        for scale_param_list in self.scale_param:
            for scale_param in scale_param_list:
                scale_param = scale_param.to(self.qkv.weight.device)

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)

        logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        #if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
        #    load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        #else:
        #    load = self._gates_to_load(gates)
        return gates, None #, load

    def forward(self, x, attn_mask=None, expert_id=0, register_hook=False, prompt=None, get_input_matrix = False):
        
        # x [Batch Size, Seq len, Dim]

        if get_input_matrix:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]
            
        B, N, C = x.shape

        q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)

        if self.apply_lora:
            k_weight = k_weight + self.lora_B_k.weight @ self.lora_A_k.weight
            v_weight = v_weight + self.lora_B_v.weight @ self.lora_A_v.weight
        
        qkv = F.linear(x, torch.cat([q_weight, k_weight, v_weight], dim=0), self.qkv.bias.data).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        k = k.permute(0, 2, 1, 3).view(x.shape[0], x.shape[1], -1)
        v = v.permute(0, 2, 1, 3).view(x.shape[0], x.shape[1], -1)

        x_re = k[:, 0, :]

        gates, load = self.noisy_top_k_gating(x_re, self.training, self.router, 
                                            self.w_noise) # hardcoded, task_id = 0

        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(k.reshape(k.shape[0], -1))

        expert_outputs = [self.experts[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                    x.shape[1], x.shape[2]).to(x))
                        for i in range(self.experts_num)]

        expert_outputs = [out.view(out.shape[0], -1) for out in expert_outputs if out.shape[0] > 0]

        y = dispatcher.combine(expert_outputs)
        k = y.view(x.shape[0], x.shape[1], x.shape[2])

        x_re = v[:, 0, :]

        gates, load = self.noisy_top_k_gating(x_re, self.training, self.router, 
                                            self.w_noise) # hardcoded, task_id = 0

        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(v.reshape(x.shape[0], -1))

        expert_outputs = [self.experts[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                    x.shape[1], x.shape[2]).to(x))
                        for i in range(self.experts_num)]

        expert_outputs = [out.view(out.shape[0], -1) for out in expert_outputs if out.shape[0] > 0]

        y = dispatcher.combine(expert_outputs)
        v = y.view(x.shape[0], x.shape[1], x.shape[2])

        k = k.view(x.shape[0], x.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(x.shape[0], x.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask.unsqueeze(0) # For head axis broadcasting

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Mlp
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Blocks
class ResidualAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_head: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_layer = MultiHeadAttention,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None,
                 text_or_image=None,
                 # For attn_layer = MultiHeadAttention_LoRA
                 lora_rank: int = 0,
                 lora_bias: bool = False
    ):
        super().__init__()

        if attn_layer == MultiHeadAttention:
            self.attn = attn_layer(d_model, n_head, qkv_bias, qk_scale, attn_drop, proj_drop)
        elif attn_layer == MultiHeadAttention_LoRA:
            self.attn = attn_layer(d_model, n_head, qkv_bias, qk_scale, attn_drop, proj_drop, lora_rank, lora_bias)
        elif attn_layer == MultiHeadAttention_MaskedLoRA:
            self.attn = attn_layer(d_model, n_head, qkv_bias, qk_scale, attn_drop, proj_drop, lora_rank, lora_bias)
        elif attn_layer == MultiHeadAttention_MoEMaskedLoRA:
            self.attn = attn_layer(d_model, n_head, qkv_bias, qk_scale, attn_drop, proj_drop, lora_rank, lora_bias)
        else:
            assert 0, f'{attn_layer} not Implemented'
            
        self.ln_1 = norm_layer(d_model)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), act_layer=act_layer)
        self.ln_2 = norm_layer(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_mask = attn_mask
        self.text_or_image = text_or_image
    
    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = self.attn_mask.to(x) if self.attn_mask is not None else None
    
        x = x.permute(1, 0, 2)
        attn = self.attn(x, attn_mask=self.attn_mask, **kwargs)
        attn = attn.permute(1, 0, 2)

        return attn

    def forward(self, x: torch.Tensor, **kwargs):
        
        x = x + self.drop_path(self.attention(self.ln_1(x), **kwargs)) # [Seq, Batch, Dim]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return x

class ResidualAttentionBlock_MLP(ResidualAttentionBlock):
    def __init__(self, 
                 d_model: int, 
                 n_head: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_layer = MultiHeadAttention, 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None, 
                 text_or_image=None,
                 # For attn_layer = MultiHeadAttention_LoRA
                 lora_rank: int = 0,
                 lora_bias: bool = False,
    ):
        super().__init__(
            d_model,
            n_head,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            drop_path,
            attn_layer, 
            act_layer,
            norm_layer,
            attn_mask, 
            text_or_image)

        self.ffn_num = 64
        self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                init_option='lora', adapter_scalar=0.1, adapter_layernorm_option='none')

        self.lora_feature = None # Temporary save the output of adapter, for method : DMNSP
    

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = self.attn_mask.to(x) if self.attn_mask is not None else None
    
        x = x.permute(1, 0, 2)
        attn = self.attn(x, attn_mask=self.attn_mask, **kwargs)
        attn = attn.permute(1, 0, 2)

        return attn

    def forward(self, x: torch.Tensor, compute_lora_feat = False, **kwargs):
        
        x = x + self.drop_path(self.attention(self.ln_1(x), **kwargs)) # [Seq, Batch, Dim]

        x_re = x.permute(1, 0, 2)
        adapt_x = self.adaptmlp(x_re, add_residual=False)
        adapt_x = adapt_x.permute(1, 0, 2)

        x = x + self.drop_path(self.mlp(self.ln_2(x)) + adapt_x)

        if compute_lora_feat:
            self.lora_feature = adapt_x.detach().cpu()

        return x

class ResidualAttentionBlock_MaskedMLP(ResidualAttentionBlock):
    def __init__(self, 
                 d_model: int, 
                 n_head: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_layer = MultiHeadAttention, 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None, 
                 text_or_image=None,
                 # For attn_layer = MultiHeadAttention_LoRA
                 lora_rank: int = 0,
                 lora_bias: bool = False,
    ):
        super().__init__(
            d_model,
            n_head,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            drop_path,
            attn_layer, 
            act_layer,
            norm_layer,
            attn_mask, 
            text_or_image)

        self.ffn_num = 64
        self.adaptmlp = MaskedAdapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                init_option='lora', adapter_scalar=0.1, adapter_layernorm_option='none')

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = self.attn_mask.to(x) if self.attn_mask is not None else None
    
        x = x.permute(1, 0, 2)
        attn = self.attn(x, attn_mask=self.attn_mask, **kwargs)
        attn = attn.permute(1, 0, 2)

        return attn

    def forward(self, x: torch.Tensor, compute_input_matrix = False, **kwargs):
        
        x = x + self.drop_path(self.attention(self.ln_1(x), **kwargs)) # [Seq, Batch, Dim]

        x_re = x.permute(1, 0, 2)
        adapt_x = self.adaptmlp(x_re, add_residual=False, compute_input_matrix=compute_input_matrix)
        adapt_x = adapt_x.permute(1, 0, 2)

        x = x + self.drop_path(self.mlp(self.ln_2(x)) + adapt_x)

        return x


class ResidualAttentionBlock_MoE_MLP(ResidualAttentionBlock):
    def __init__(self, 
                 d_model: int, 
                 n_head: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_layer = MultiHeadAttention, 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None, 
                 text_or_image=None,
                 # For attn_layer = MultiHeadAttention_LoRA
                 lora_rank: int = 0,
                 lora_bias: bool = False,
                 # MoE
                 step: int = 0,
                 experts_num: int = 0,
                 top_k: int = 0,
                 noisy_gating: bool = True
    ):
        super().__init__(
            d_model,
            n_head,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            drop_path,
            attn_layer, 
            act_layer,
            norm_layer,
            attn_mask, 
            text_or_image)

        assert top_k <= experts_num

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.step = step
        self.top_k = top_k
        self.noisy_gating = noisy_gating

        self.ffn_num = 64
        self.experts_num = experts_num
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        
        self.router_list = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True) for _ in range(self.step)
        ])
        self.w_noise_list = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True) for _ in range(self.step)
        ])

        self.adaptmlp_list = nn.ModuleList([
            Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=0.1,
                                    adapter_layernorm_option='none')
            for _ in range(self.experts_num)                           
        ])

        self.lora_feature = None # Temporary save the output of adapter, for method : DMNSP
        
    
    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    
        x = x.permute(1, 0, 2)
        attn = self.attn(x, attn_mask=self.attn_mask, **kwargs)
        attn = attn.permute(1, 0, 2)

        return attn

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        #if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
        #    load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        #else:
        #    load = self._gates_to_load(gates)
        return gates, None #, load

    def forward(self, x: torch.Tensor, compute_lora_feat=False, **kwargs):
        
        x = x + self.drop_path(self.attention(self.ln_1(x), **kwargs)) # [Seq, Batch, Dim]

        x_re = x.permute(1, 0, 2)[:, 0, :]
        gates, load = self.noisy_top_k_gating(x_re, self.training, self.router_list[0], 
                                            self.w_noise_list[0]) # hardcoded, task_id = 0
        
        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))

        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                    x.shape[0], x.shape[2]).to(x), add_residual=False)
                        for i in range(self.experts_num)]

        expert_outputs = [out.view(out.shape[0], -1) for out in expert_outputs if out.shape[0] > 0]

        y = dispatcher.combine(expert_outputs)
        y = y.view(x.shape[1], x.shape[0], x.shape[2])
        x = x + self.drop_path(self.mlp(self.ln_2(x)) + y.permute(1, 0, 2))

        return x

class ResidualAttentionBlock_MoE_Proj(ResidualAttentionBlock):
    def __init__(self, 
                 d_model: int, 
                 n_head: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 drop_path: float = 0.,
                 attn_layer = MultiHeadAttention, 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None, 
                 text_or_image=None, 
                 # For attn_layer = MultiHeadAttention_LoRA
                 lora_rank: int = 0,
                 lora_bias: bool = False,
                 # MoE
                 experts_num=0, 
    ):
        super().__init__()

        if isinstance(attn_layer, str):
            try:
                attn_layer = globals()[attn_layer]
            except KeyError:
                print(f'{attn_layer} not found, using default MultiHeadAttention')
                attn_layer = MultiHeadAttention

        if isinstance(act_layer, str):
            try:
                act_layer = globals()[act_layer]
            except KeyError:
                print(f'{act_layer} not found, using default nn.GELU')
                act_layer = nn.GELU
                
        if isinstance(norm_layer, str):
            try:
                norm_layer = globals()[norm_layer]
            except KeyError:
                print(f'{norm_layer} not found, using default nn.LayerNorm')
                norm_layer = nn.LayerNorm

        self.attn = attn_layer(d_model, n_head, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.ln_1 = norm_layer(d_model)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), act_layer=act_layer)
        self.ln_2 = norm_layer(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_mask = attn_mask
        self.is_train = True
        # TODO : make it argument, now harcodrd
        if experts_num > 1:
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            self.step = 1
        else:
            self.step = 0
        self.top_k = 2
        self.ffn_num = 64
        self.experts_num = experts_num
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.noisy_gating = True
        self.text_or_image = text_or_image
        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()

        for i in range(self.step):
            self.router_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
            self.w_noise_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
        
        self.adaptmlp_list = nn.ModuleList()
        for i in range(self.experts_num):  #
            self.adaptmlp_list.append(Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=0.1,
                                    adapter_layernorm_option='none',
                                    ))

        self.lora_feature = None # Temporary save the output of adapter, for method : DMNSP
    
    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    
        x = x.permute(1, 0, 2)
        attn = self.attn(x, attn_mask=self.attn_mask, **kwargs)
        attn = attn.permute(1, 0, 2)

        return attn

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        #if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
        #    load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        #else:
        #    load = self._gates_to_load(gates)
        return gates, None #, load

    def forward(self, x: torch.Tensor, **kwargs):
        
        x = x + self.drop_path(self.attention(self.ln_1(x), **kwargs)) # [Seq, Batch, Dim]

        if self.experts_num == 0:

            x = x + self.drop_path(self.mlp(self.ln_2(x)))

        elif self.experts_num == 1:

            x_re = x.permute(1, 0, 2)
            adapt_x = self.adaptmlp_list[0](x_re, add_residual=False)
            adapt_x = adapt_x.permute(1, 0, 2)

            x = x + self.drop_path(self.mlp(self.ln_2(x)) + adapt_x)

            if compute_lora_feat:
                self.lora_feature = adapt_x.detach().cpu()

        else:

            x_re = x.permute(1, 0, 2)[:, 0, :]
            gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router_list[0], 
                                                self.w_noise_list[0]) # hardcoded, task_id = 0
            
            dispatcher = SparseDispatcher(self.experts_num, gates)
            expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))

            expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                        x.shape[0], x.shape[2]).to(x), add_residual=False)
                            for i in range(self.experts_num)]

            expert_outputs = [out.view(out.shape[0], -1) for out in expert_outputs if out.shape[0] > 0]

            y = dispatcher.combine(expert_outputs)
            y = y.view(x.shape[1], x.shape[0], x.shape[2])
            x = x + self.drop_path(self.mlp(self.ln_2(x)) + y.permute(1, 0, 2))

        return x

# Transformers
class Transformer(nn.Module):
    def __init__(self, 
                 width: int, 
                 layers: int, 
                 heads: int, 
                 block_layer = ResidualAttentionBlock,
                 attn_layer = MultiHeadAttention, 
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 attn_mask: torch.Tensor = None, 
                 text_or_image=None,
                 **kwargs
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        if isinstance(block_layer, str):
            try:
                block_layer = globals()[block_layer]
            except KeyError:
                print(f'{block_layer} not found, using default ResidualAttentionBlock')
                block_layer = ResidualAttentionBlock

        if isinstance(attn_layer, str):
            try:
                attn_layer = globals()[attn_layer]
            except KeyError:
                print(f'{attn_layer} not found, using default MultiHeadAttention')
                attn_layer = MultiHeadAttention

        if isinstance(act_layer, str):
            try:
                act_layer = globals()[act_layer]
            except KeyError:
                print(f'{act_layer} not found, using default nn.GELU')
                act_layer = nn.GELU
                
        if isinstance(norm_layer, str):
            try:
                norm_layer = globals()[norm_layer]
            except KeyError:
                print(f'{norm_layer} not found, using default nn.LayerNorm')
                norm_layer = nn.LayerNorm

        self.blocks = nn.ModuleList([
            block_layer(
                d_model=width, 
                n_head=heads, 
                attn_layer=attn_layer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_mask=attn_mask, 
                text_or_image=text_or_image, 
                **kwargs) 
                for _ in range(layers)])

    def forward(self, x: torch.Tensor, **kwargs):
        
        for block in self.blocks:
            x = block(x, **kwargs)
            
        return x

# ViT from CLIP
class VisualTransformer(nn.Module):
    def __init__(self, 
                 img_size: int, 
                 patch_size: int,
                 in_chans: int = 3, 
                 width: int = 768, 
                 depth: int = 12, 
                 heads: int = 8, 
                 output_dim: int = 512, 
                 text_or_image: str = None,
                 **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.width = width
        self.depth = depth
        self.heads = heads
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, depth, heads, text_or_image=text_or_image, **kwargs)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, **kwargs):

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND / [Batch_Size, Seq_len, Dim] -> [Seq_len, Batch_Size, Dim]
        x = self.transformer(x, **kwargs)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

# Standard ViT
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 attn_layer=MultiHeadAttention, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 representation_size=None,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 ckpt_layer=0,
                 **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        '''
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model = embed_dim,
                n_head = num_heads,
                mlp_ratio = mlp_ratio, 
                qkv_bias = qkv_bias, 
                qk_scale = qk_scale,
                attn_drop = attn_drop_rate,
                proj_drop = drop_rate,
                drop_path = dpr[i],
                attn_layer = attn_layer,
                act_layer = nn.GELU,
                norm_layer = norm_layer
            )
            for i in range(depth)])
        '''

        self.transformer = Transformer(embed_dim, depth, num_heads, text_or_image='image', attn_layer=attn_layer, norm_layer=norm_layer, **kwargs)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, task_id=-1, **kwargs):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        # TODO: clean, move everything to trasnformer
        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        if prompt is not None:
            for i,blk in enumerate(self.transformer.blocks):

                if prompt is not None:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id)
                else:
                    p_list = None

                # the blk only takes x in shape [N, B, C] not [B, N ,C]
                x = x.permute(1, 0, 2)
                x = blk(x, register_hook=register_blk==i, prompt=p_list, **kwargs)
                x = x.permute(1, 0, 2)
        else:

            x = x.permute(1, 0, 2)
            x = self.transformer(x, **kwargs)
            x = x.permute(1, 0, 2)

        x = self.norm(x)
        
        return x, prompt_loss

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
