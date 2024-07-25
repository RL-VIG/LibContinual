from typing import Tuple, Literal

import torch
import torch.nn as nn

from timm.models.layers import PatchEmbed

from einops import rearrange

from transformers import ViTModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class PreNorm(nn.Module):

    # Pre-normalization layer

    def __init__(self,
                 dim: int,
                 fn: nn.Module # feed-forward network
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):

    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 drop_rate: float = 0.
    ):            
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate)
        )

    
    def forward(self, x):
        return self.net(x)

# Attention in Vision Transformer
class Attention(nn.Module):

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 drop_rate: float = 0.
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(drop_rate)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Attention_LoRA(Attention):

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64, 
                 drop_rate: float = 0.,
                 r: int = 8, # 降维的目标维度，升维的原维度
                 num_branches: int = 1 # 多少个分支，每个分支降维到r维度，升维到原维度，在 inflora 中等于 n_tasks


    ):
        super().__init__(dim, heads, dim_head, drop_rate)

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(num_branches)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(num_branches)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(num_branches)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(num_branches)])


        self.layers = nn.Sequential()
        for _ in num_branches:
            self.layers.append(nn.Linear(dim, r))


    def forward(self, x: torch.Tensor):
        x = self.layers(x)

        for _ in tasks: # how many tasks then how many a,b
            x += ...

        pass

class Transformer(nn.Module):

    def __init__(self,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 0,
                 dim: int = 2048,
                 heads: int = 8,
                 dim_heads: int = 64,
                 drop_rate: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.layers.append(PreNorm(dim, Attention(dim, heads, dim_heads, drop_rate)))
            self.layers.append(PreNorm(dim, FeedForward()))

        for _ in range(num_decoder_layers):    
            # To implement    
            pass


    def forward(self, x):
        for layer in self.layers:
            x += layer(x)
        return x

class VisionTransformer(nn.module):

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),  # input image size
                 patch_size: int = 16, # patch size
                 num_channel: int = 3,  # number of input channels
                 num_classes: int = 1000,  # number of output classes
                 glopal_pool: Literal['mean', 'cls'] = 'cls', # cls token or mean pooling
                 embed_dim: int = 768,  # embedding dimension
                 drop_rate: float = 0. # dropout rate
    ):
        super().__init__()

        self.to_patch_embedding = PatchEmbed(img_size, patch_size, num_channel, embed_dim) # patch to embedding
        self.num_patches = self.to_patch_embedding.num_patches

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) # 1 for cls token?
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(drop_rate)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        ) # dropout?

        # -------

        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        lora_config = LoraConfig(
            target_modules = [], # modules to add LoRA modules and train the LoRA modules
            modules_to_save = [], # modules to train


            inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )


        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()




    def forward(self, x):
        x = self.to_patch_embedding(x)

        # --------

        return self.model(x)

        pass