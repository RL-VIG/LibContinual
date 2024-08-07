import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from peft import get_peft_model, LoraConfig
from core.model.backbone.vit import Attention
from sklearn.cluster import KMeans

def dispatch_inflora(target: torch.nn.Module,
                adapter_name: str,
                **kwargs
    ):

        new_module = Attention_InfLoRA(target, adapter_name, **kwargs)

        return new_module
        
class Attention_InfLoRA(nn.Module):
    def __init__(self,
                 target: Attention,
                 adapter_name: str,
                 r: int = 10,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 init_lora_weights: bool = True,
                 use_rslora: bool = False,
                 n_tasks: int = 10,
                 **kwargs
                ):
        super().__init__()
        self.dim = target.dim
        self.num_heads = target.num_heads
        self.scale = target.scale
        self.qkv = target.qkv
        self.attn_drop = target.attn_drop
        self.proj = target.proj
        self.proj_drop = target.proj_drop
        self.attn_gradients = target.attn_gradients
        self.attention_map = target.attention_map

        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_alpha = lora_alpha

        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(self.dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, self.dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(self.dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, self.dim, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(self.dim ,self.dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(self.dim ,self.dim)
        self.n_cur_matrix = 0

        self.weight = self.qkv.weight

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x, register_hook=False, update_cur_feat = False, task_id = -1, **kwargs):
        if update_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]
            return 0

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task_id > -1:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task_id+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task_id+1)], dim=0).sum(dim=0)
            
            # Apply lora_alpha
            weight_k = weight_k * self.lora_alpha
            weight_v = weight_v * self.lora_alpha

            # Apply lora_dropout
            weight_k = self.lora_dropout(weight_k)
            weight_v = self.lora_dropout(weight_v)
            
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

        self._cur_task_id = -1
        self._peft_implementation = True

        if self._peft_implementation:
            lora_config = LoraConfig(
                target_modules = [n for n, m in backbone.named_modules() if isinstance(m, Attention)], # modules to add LoRA modules and train the LoRA modules
                inference_mode=False, 
                r = kwargs["rank"], 
                lora_alpha = 1, 
                lora_dropout = 0.0
            )
            lora_config._register_custom_module({
                    # Usually replace Linear layers for minor change to the whole structure
                    Attention : dispatch_inflora
                }) # specifying replace key class modules with value class modules
            self.backbone = get_peft_model(backbone, lora_config)
        else:
            self.backbone = backbone

        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs["init_cls_num"], bias=True)
            for _ in range(kwargs["task_num"])
        ])

    def update_fc(self):
        self._cur_task_id += 1

    def get_feature(self, x):
        features, prompt_loss = self.backbone(x, task_id = self._cur_task_id)
        return features

    def forward(self, x, inference = False):
        logits = []
        features, prompt_loss = self.backbone(x, task_id = self._cur_task_id)
        if inference:
            for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                logits.append(prompts(features))
        else:
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits.append(prompts(features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': features,
            'prompt_loss': prompt_loss
        }

    def update_cur_feat(self, x):
        self.backbone(x, task_id = self._cur_task_id, update_cur_feat = True)

class InfLoRA(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        # Initialize some variables here
        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]
        
        self.topk = 1
        self._known_classes = 0
        self.all_keys = []
        self.feature_list = []
        self.project_type = []

        # Sinet contain ViT as image encoder (backbone) and classifier pool
        self.model = Sinet(backbone, **kwargs)

        # Initialize parameters of LoRA modules
        for module in self.model.modules():
            if isinstance(module, Attention_InfLoRA):
                module.init_param()

        self.model.to(self.device)

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        if task_idx == 1:
            self._known_classes = self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num
        self.model.update_fc()

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "classifier_pool." + str(task_idx) in name:
                param.requires_grad_(True)
            if "lora_B_k." + str(task_idx) in name:
                param.requires_grad_(True)
            if "lora_B_v." + str(task_idx) in name:
                param.requires_grad_(True)
    
        unfrezeed_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                unfrezeed_params.append(name)

        print("Current task : " + str(task_idx) + ", Parameters to be updated: " + str(len(unfrezeed_params)))
        print(",".join(unfrezeed_params))

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch['image'].to(self.device)
                self.model.update_cur_feat(inputs)

            # Initialize LoRA A
            if task_idx == 0: # first task
                for module in self.model.backbone.modules():
                    if isinstance(module, Attention_InfLoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[task_idx].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[task_idx].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else: 
                kk = 0
                for module in self.model.backbone.modules():
                    if isinstance(module, Attention_InfLoRA):
                        cur_matrix = module.cur_matrix
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                        elif self.project_type[kk] == 'retain':
                            cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                        else:
                            raise ValueError(f'project_type should be remove or retain, not {self.project_type[kk]}')

                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[task_idx].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[task_idx].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

    def observe(self, data):

        # Mask learned classes, leaving only unlearned classes 
        inputs, targets = data['image'].to(self.device), data['label'].to(self.device)
        mask = (targets >= self._known_classes).nonzero().view(-1)
        inputs = torch.index_select(inputs, 0, mask)
        targets = torch.index_select(targets, 0, mask) - self._known_classes

        logits = self.model(inputs)['logits']
        loss = F.cross_entropy(logits, targets)

        _, preds = torch.max(logits, dim=1)
        correct = preds.eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return preds, acc, loss

    def inference(self, data):

        inputs, targets = data['image'].to(self.device), data['label']
        logits = self.model(inputs, inference = True)['logits']
        _, preds = torch.max(logits, dim=1)

        correct = preds.cpu().eq(targets.expand_as(preds)).sum().item()
        total = len(targets)

        acc = round(correct / total, 4)

        return logits, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch['image'].to(self.device)
                self.model.update_cur_feat(inputs)

            mat_list = []
            for module in self.model.backbone.modules():
                if isinstance(module, Attention_InfLoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            # self.update_GPM(mat_list)
            self.update_DualGPM(mat_list, task_idx)

            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf=torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                self.feature_mat.append(Uf)

        # TODO: clustering
        self.clustering(train_loader)

    def clustering(self, dataloader):
        features = []

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)

            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)

            with torch.no_grad():
                feature = self.model.get_feature(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def update_DualGPM (self, mat_list, task_idx):
        threshold = (self.lame - self.lamb)*task_idx/self.task_num + self.lamb
        print ('Threshold: ', threshold) 
        if task_idx == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)

    def get_parameters(self, config):
        return self.model.parameters()