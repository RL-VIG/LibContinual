import torch
import torch.nn as nn
import copy

from core.model.backbone.vit_official import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from transformers import ViTModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.tuners.lora.layer import LoraLayer

class Model(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")    

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
            nn.Linear(args["embd_dim"], self.class_num, bias=True) for i in range(args["total_sessions"])
        ])

class SiNet(nn.Module):
        
    def __init__(self, args):
        super().__init__()

        self.num_task = 0 # total number of tasks observed so far

        self.image_encoder = None

        self.model = Model(args)




    def update_task(self):
        self.num_task += 1


class InfLoRA():

    # init, observe, inference, after_task, before_task, get_parameteres?

    def __init__(self, backbone, device, **kwargs):
        self.device = device

        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]


        self.cur_task = -1 # current task id

        self.known_classes = 0 # total number of classes learned so far
        self.total_classes = 0 # total number of classes observed so far
        




        
    
    def observe(self, data):

        self.cur_task += 1
        self.model.update_task()
        if (self.cur_task == 0):
            self.total_classes += self.init_cls_num
        else: 
            self.total_classes += self.inc_cls_num
        
        # if multiple gpu, Dataparallel should be used here
        
        # train and clustering
        # self.to(self.device) # called everytime before training?
        

        # I dont see any parameter with these name in model
        # What if the dot symbol means extracing element from list named classifier_pool?
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            # distributed training is not supported yet
            if "classifier_pool" + "." + str(self.model.num_task - 1) in name:
                param.requires_grad_(True)
            if "lora_B_k" + "." + str(self.model.num_task - 1) in name:
                param.requires_grad_(True)
            if "lora_B_v" + "." + str(self.model.num_task - 1) in name:
                param.requires_grad_(True)

        to_update_param = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                to_update_param.add(name)

        with torch.no_grad():

            data = data.to(self.device)
            # ! need to check format of data
            self.model(data, get_cur_feat=True)

            if self.cur_task == 0:
                for module in self.model.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                # kk = 0
                # for module in self._network.modules():
                #     if isinstance(module, Attention_LoRA):
                #         cur_matrix = module.cur_matrix
                #         cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                #         cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                #         module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.cur_matrix.zero_()
                #         module.n_cur_matrix = 0
                #         kk += 1

                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                        else:
                            assert self.project_type[kk] == 'retain'
                            cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1