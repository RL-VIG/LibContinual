import torch
import torch.nn as nn

from peft import get_peft_model, LoraConfig

class Model(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone

        print(self.backbone.modules)
        assert(0)

        lora_config = LoraConfig(
            target_modules = [], # modules to add LoRA modules and train the LoRA modules
            modules_to_save = [], # modules to train
            _custom_modules = {
                torch.nn.Linear : InfLoRA
            }, # custom modules to add to the model
            inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

class Sinet(nn.Module):
    def __init__(self, args):
        super.__init__()




    def forward():
        pass