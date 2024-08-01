import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig
from core.model.backbone.vit import Attention

class Attention_InfLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Sinet(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()

        self._num_task = 0 # ⁡⁢⁣⁢What for, it is update in function: update_fc⁡

        lora_config = LoraConfig(
            target_modules = [n for n, m in backbone.named_modules() if isinstance(m, Attention)], # modules to add LoRA modules and train the LoRA modules
            modules_to_save = [], # modules to train
            _custom_modules = {
                # Usually replace Linear layers for minor change to the whole structure
                Attention : Attention_InfLoRA
            }, # specifying replace key class modules with value class modules
            inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        self.backbone = get_peft_model(backbone, lora_config)
        
        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], self.class_num, bias=True)
            for _ in range(kwargs["total_sessions"])
        ])

    def update_fc(self):
        self._num_task += 1

class InfLoRA(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        # Initialize some variables here
        self.device = device

        self._cur_task = -1
        self._total_classes = 0
        self._known_classes = 0

        # Sinet contain ViT as image encoder and classifier pool
        self.model = Sinet(backbone, **kwargs)

        # Initialize parameters of LoRA modules
        for module in self.model.modules():
            if isinstance(module, Attention_InfLoRA):
                module.init_param()

    def observe(self, data):
        self._cur_task += 1
        self._total_classes += data.num_classes # TODO: num class of this data

        # TODO: get train and test loader
        self.train_loader = None
        self.test_loader = None

        # TODO: train(train and test loader) and clustering(train loader)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):

        # TODO: check if this is needed
        self.model.to(self.device)

        for name, param in self.model.named_parameters():
            print(f"name: {name}")
        
        assert(0)
        
        pass

    def inference(self, data):
        pass

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        pass

    def get_parameters(self, config):
        return self.model.parameters()