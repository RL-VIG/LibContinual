"""
@misc{wu2025sdlorascalabledecoupledlowrank,
      title={SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning}, 
      author={Yichen Wu and Hongming Piao and Long-Kai Huang and Renzhen Wang and Wanhua Li and Hanspeter Pfister and Deyu Meng and Kede Ma and Ying Wei},
      year={2025},
      eprint={2501.13198},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.13198}, 
}

Adapted from https://github.com/WuYichen-97/SD-Lora-CL

"""

import torch
import torch.nn as nn
import copy
import numpy as np

from torch.nn import functional as F
from .backbone.transformer import MultiHeadAttention_SDLoRA

class Model(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.device = device
        self.embed_dim = kwargs["embd_dim"]
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']

    def update_fc(self):
        
        self._cur_task_id += 1
        if self._cur_task_id == 0:
            classifier = nn.Linear(self.embed_dim, self.init_cls_num, bias=True)

            nn.init.kaiming_uniform_(classifier.weight, nonlinearity='linear')
            nn.init.constant_(classifier.bias, 0)
        else:
            classifier = nn.Linear(self.embed_dim, self.init_cls_num + self.inc_cls_num * (self._cur_task_id), bias=True)

            nn.init.kaiming_uniform_(classifier.weight, nonlinearity='linear')
            nn.init.constant_(classifier.bias, 0)

            nb_output = self.classifier.out_features
            classifier.weight.data[:nb_output] = copy.deepcopy(self.classifier.weight.data)
            classifier.bias.data[:nb_output] = copy.deepcopy(self.classifier.bias.data)
            del self.classifier

        self.classifier = classifier

    def forward(self, x, inference = False):

        features = self.backbone(x)        
        logits = self.classifier(features)
        return logits

class SD_LoRA(nn.Module):

    def __init__(self, backbone, device, **kwargs):

        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.init_mag = kwargs['init_mag']
        self.rank_reduction = kwargs['rank_reduction']
        self.knowledge_dist = kwargs['knowledge_dist']
        self._known_classes = 0

        self._network = Model(backbone, device, **kwargs)
        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_SDLoRA)]

    def observe(self, data):
        
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self._network(x)

        # Masked previous classes
        fake_y = y - self._known_classes
        loss = F.cross_entropy(logits[:, self._known_classes:], fake_y)

        preds = logits.max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc, loss
    
    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        logits = self._network(x, inference = True)
        preds = logits.max(1)[1]

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self._network.update_fc()

        if self.rank_reduction[0]:
            if task_idx == self.rank_reduction[1]:
                for module in self.attention_modules:
                    module.lora_rank = self.rank_reduction[3]

            elif task_idx == self.rank_reduction[2]:
                for module in self.attention_modules:
                    module.lora_rank = self.rank_reduction[4]
        
        # All blocks share same magnitude
        mag = nn.ParameterList([nn.Parameter(torch.Tensor([self.init_mag])) for _ in range(task_idx + 1)])
        for module in self.attention_modules:
            module.mag_lora = mag
            module.init_param()

        self._network = self._network.to(self.device)

        unfrezeed_params = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if f'classifier' in name or \
               f'lora' and f'list.{task_idx}' in name or \
               ('mag' in name and 'assimilated' not in name):
                param.requires_grad_(True)
                unfrezeed_params.append(name)

        print(f"Current task : {task_idx}, Parameters to be updated: {len(unfrezeed_params)}")

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

        if self.knowledge_dist[0] and task_idx > 0:
            for layer, module in enumerate(self.attention_modules):

                dirs_q, dirs_v = [], []
                for i in range(len(module.lora_A_q_list)):

                    norm_B = torch.norm(module.lora_B_q_list[i].weight)
                    norm_A = torch.norm(module.lora_A_q_list[i].weight)

                    if norm_A != 0 and norm_B != 0:
                        dirs_q.append(
                            (module.lora_B_q_list[i].weight @ module.lora_A_q_list[i].weight) / (norm_B * norm_A)
                        )
                    else: # zero-tensor, for consistency
                        dirs_q.append(
                            (module.lora_B_q_list[i].weight @ module.lora_A_q_list[i].weight)
                        )

                    norm_B = torch.norm(module.lora_B_v_list[i].weight)
                    norm_A = torch.norm(module.lora_A_v_list[i].weight)

                    if norm_A != 0 and norm_B != 0:
                        dirs_v.append(
                            (module.lora_B_v_list[i].weight @ module.lora_A_v_list[i].weight) / (norm_B * norm_A)
                        )
                    else: # zero-tensor, for consistency
                        dirs_v.append(
                            (module.lora_B_q_list[i].weight @ module.lora_A_q_list[i].weight)
                        )

                flatten_dirs = [dir_q.flatten() for dir_q in dirs_q]

                last_dir = flatten_dirs[-1].unsqueeze(1)
                prev_dirs = torch.stack(flatten_dirs[:-1], dim=-1)

                alphas = torch.linalg.lstsq(prev_dirs, last_dir)

                if alphas.residuals < self.knowledge_dist[1]:
                    print(f'Layer {layer}: {alphas.residuals.item()} < {self.knowledge_dist[1]}, Q Merged with {alphas.solution}')

                    assert prev_dirs.shape[1] == len(module.assimilated_mag_lora_q) - 1
                    for ii in range(prev_dirs.shape[1]):
                        module.assimilated_mag_lora_q[ii] += alphas.solution[i]

                    nn.init.zeros_(module.lora_B_q_list[task_idx])
                    nn.init.zeros_(module.lora_A_q_list[task_idx])

                flatten_dirs = [dir_v.flatten() for dir_v in dirs_v]

                last_dir = flatten_dirs[-1].unsqueeze(1)
                prev_dirs = torch.stack(flatten_dirs[:-1], dim=-1)

                alphas = torch.linalg.lstsq(prev_dirs, last_dir)

                if alphas.residuals < self.knowledge_dist[1]:
                    print(f'Layer {layer}: {alphas.residuals.item()} < {self.knowledge_dist[1]}, V Merged with {alphas.solution}')

                    assert prev_dirs.shape[1] == len(module.assimilated_mag_lora_v) - 1
                    for ii in range(prev_dirs.shape[1]):
                        module.assimilated_mag_lora_v[ii] += alphas.solution[i]

                    nn.init.zeros_(module.lora_B_v_list[task_idx])
                    nn.init.zeros_(module.lora_A_v_list[task_idx])

    def get_parameters(self, config):
        return self._network.parameters()