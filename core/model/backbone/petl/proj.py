import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Proj(nn.Module):
    def __init__(self,
                 d_model=None,
                 id=-1):
        super().__init__()

        self.eye = nn.Parameter(torch.eye(d_model))

        self.space = [torch.tensor((1)), torch.tensor((1))]
        self.scale_param = nn.ParameterList([nn.Parameter(self.eye) for _ in range(2)])
        self.scaling_mask = [False, False]
        self.id = -1

    def forward(self, x, kv_w, expert_id):

        if expert_id == self.id:
            pass
        else:
            return F.linear(x, kv_w)

        pre_kv_w = None

        for mask, scale, space in zip(self.scaling_mask, self.scale_param, self.space):

            if not mask:
                break

            scale_size = space.shape[1]
            cropped_scale = scale[:scale_size, :scale_size]

            cropped_scale = cropped_scale @ cropped_scale.T # better, idk why

            cropped_identity_matrix = self.eye[:scale_size, :scale_size].to(x)

            if pre_kv_w is None:
                pre_kv_w = kv_w + kv_w @ space @ (cropped_scale - cropped_identity_matrix) @ space.T
            else:
                pre_kv_w = pre_kv_w + pre_kv_w @ space @ (cropped_scale - cropped_identity_matrix) @ space.T

        if pre_kv_w is None:
            return F.linear(x, kv_w)
        else:
            return F.linear(x, pre_kv_w)

class Proj2(nn.Module):
    def __init__(self,
                 d_model=None,
                 id=-1):
        super().__init__()

        self.eye = nn.Parameter(torch.eye(d_model))

        self.space = [torch.tensor((1)), torch.tensor((1))]
        self.scale_param = nn.ParameterList([nn.Parameter(self.eye) for _ in range(2)])
        self.scaling_mask = [False, False]
        self.id = -1

    def forward(self, x, kv_w, expert_id):

        if expert_id == self.id:
            pass
        else:
            return F.linear(x, kv_w)

        pre_kv_w = None

        for mask, scale, space in zip(self.scaling_mask, self.scale_param, self.space):

            if not mask:
                break

            scale_size = space.shape[1]
            cropped_scale = scale[:scale_size, :scale_size]

            cropped_scale = cropped_scale @ cropped_scale.T # better, idk why

            cropped_identity_matrix = self.eye[:scale_size, :scale_size].to(x)

            if pre_kv_w is None:
                pre_kv_w = kv_w + kv_w @ space @ (cropped_scale - cropped_identity_matrix) @ space.T
            else:
                pre_kv_w = pre_kv_w + pre_kv_w @ space @ (cropped_scale - cropped_identity_matrix) @ space.T

        if pre_kv_w is None:
            return F.linear(x, kv_w)
        else:
            return F.linear(x, pre_kv_w)