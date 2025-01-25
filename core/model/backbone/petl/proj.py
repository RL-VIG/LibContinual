import math
import torch
import torch.nn as nn

class Proj(nn.Module):
    def __init__(self,
                 d_model=None):
        super().__init__()

        self.eye = nn.Parameter(torch.eye(d_model))

    def forward(self, x):
        return x @ self.eye.to(x)