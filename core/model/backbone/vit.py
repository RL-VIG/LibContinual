'''
Code Reference:
Adapted from https://github.com/GT-RIPL/CODA-Prompt
'''

import os
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
from .prompt import L2P, CodaPrompt, DualPrompt
from .transformer import MultiHeadAttention_LoRA, VisionTransformer

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint
        
class ViTZoo(nn.Module):
    def __init__(self, pretrained = False, model_name='vit_base_patch16_224', attn_layer='MultiHeadAttention', **kwargs):
        super(ViTZoo, self).__init__()
        
        self.task_id = None
        self.feat_dim = 768

        self.feat = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                    num_heads=12, ckpt_layer=0,
                                    drop_path_rate=0, attn_layer=attn_layer,
                                    **kwargs
                                    )

        if pretrained:
            print(f'Using pretrained model : {model_name}')

            if model_name == 'vit_base_patch16_224.augreg2_in21k_ft_in1k' and os.path.exists('/home/lvqiexuan/.cache/torch/hub/checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k.pt'):
                # Manually Loading weight
                load_dict = torch.load('/home/lvqiexuan/.cache/torch/hub/checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k.pt')
            else:
                load_dict = timm.create_model(model_name, pretrained = pretrained).state_dict()
            
            key_mapping = {
                ".norm1.": ".ln_1.",
                ".norm2.": ".ln_2.",
                "blocks.": "transformer.blocks."
            }

            modified_load_dict = {}
            for key in load_dict.keys():
                new_key = key
                for old_key, mapped_key in key_mapping.items():
                    if old_key in new_key:
                        new_key = new_key.replace(old_key, mapped_key)

                modified_load_dict[new_key] = load_dict[key]

            self.feat.load_state_dict(modified_load_dict, strict = False)

        self.prompt = None
        self.prompt_flag = ''
        
    def create_prompt(self, prompt_flag, **kwargs):
        self.prompt_flag = prompt_flag

        if self.prompt_flag == 'l2p':
            self.prompt = L2P(**kwargs)
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, **kwargs)
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, **kwargs)
           
    # pen: get penultimate features    
    def forward(self, image, text=None, pen=False, train=False, **kwargs):

        if self.prompt_flag == 'l2p':

            with torch.no_grad():
                self.eval()
                cls_features = self.feat(image, prompt_flag = self.prompt_flag)

            if train:
                self.train()

            out, reduce_sim = self.feat(
                x = image,
                prompt = self.prompt,
                cls_features = cls_features,
                prompt_flag = self.prompt_flag
            )

            return out, reduce_sim

        elif self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(image)
                q = q[:,0,:]

            # q?, train?, task_id?
            out, prompt_loss = self.feat(image, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(image, **kwargs) 
            out = out[:,0,:]
            
        out = out.view(out.size(0), -1)

        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
            
class ViT_in21k_adapter(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(ViT_in21k_adapter, self).__init__()

        self.task_id = None
        self.feat_dim = 768
        # get feature encoder
        if pretrained:
            print("Using pretrained model")
            from core.model.backbone.petl import vision_transformer_adapter
            from easydict import EasyDict

            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=64,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )

            zoo_model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            zoo_model.out_dim=768
            zoo_model.eval()

        self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    def create_prompt(self, prompt_flag, **kwargs):
        self.prompt_flag = prompt_flag
        # self.prompt_param = prompt_param
        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, **kwargs)
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, **kwargs)
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, **kwargs)
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:,0,:]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out = self.feat(x) # This implementation of adapter vit doesn't return prompt loss
            
        out = out.view(out.size(0), -1)
        # if not pen:
        #     out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

def vit_pt_imnet(pretrained=False, **kwargs):
    return ViTZoo(pretrained, **kwargs)

def vit_pt_imnet_in21k_adapter(pretrained=False, **kwargs):
    return ViT_in21k_adapter(pretrained, **kwargs)
