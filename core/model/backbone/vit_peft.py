from typing import Tuple, Literal

import torch
import torch.nn as nn

from timm.models.layers import PatchEmbed

from einops import rearrange

from transformers import ViTModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.tuners.lora.layer import LoraLayer


class InfLoRA(torch.nn.Module, LoraLayer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()




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

        # -------

        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        lora_config = LoraConfig(
            target_modules = [], # modules to add LoRA modules and train the LoRA modules
            modules_to_save = [], # modules to train
            _custom_modules = {
                torch.nn.Linear : InfLoRA
            }, # custom modules to add to the model




            inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )


        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()




    def forward(self, x):
        x = self.to_patch_embedding(x)

        # --------

        return self.model(x)

        pass