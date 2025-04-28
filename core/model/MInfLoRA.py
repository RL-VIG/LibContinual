"""
Code Reference:
https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from .backbone.transformer import MultiHeadAttention_MaskedLoRA1

GREEDY=True
APPROX_FEAT=True
    
Epsilon = 0.5

def _set_random(seed):
    '''
    Set random values on various devices to ensure repeatable results
    '''

    seed = int(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_probe_greedy_span_unified_normalized(cur_matrixs_list, energy_threshold=0.95, top_r=None):
    """
    Greedy span selection across multiple attention blocks, with per-block normalization.
    Dynamically select samples that together span a certain percentage of gradient space.

    Args:
        cur_matrixs_list (List[torch.Tensor]): list of (Num, 768, 768) tensors for each block.
        energy_threshold (float): fraction of gradient space to cover.
        top_r (int, optional): number of top singular vectors to use.

    Returns:
        selected_indices (torch.Tensor)
    """
    N = cur_matrixs_list[0].shape[0]
    device = cur_matrixs_list[0].device

    # 1. Normalize each block independently
    normalized_cur_matrices = []
    for matrices in cur_matrixs_list:
        frob_norms = torch.norm(matrices.view(N, -1), dim=-1, p=2).view(N, 1, 1)  # (N, 1, 1)
        matrices_normalized = matrices / (frob_norms + 1e-8)
        normalized_cur_matrices.append(matrices_normalized)

    # 2. Compute global covariance C
    C_global = sum([matrices.sum(dim=0) for matrices in normalized_cur_matrices])

    # 3. SVD on global C
    U, _, _ = torch.linalg.svd(C_global)
    if top_r is not None:
        U = U[:, :top_r]

    # 4. Compute projected sample vectors
    projected_vectors = []
    for i in range(N):
        x_cov_sum = sum([matrices[i] for matrices in normalized_cur_matrices])  # (768, 768)
        proj = U.T @ x_cov_sum @ U  # (top_r, top_r)
        projected_vectors.append(proj.flatten())  # flatten to (top_r*top_r,)
    projected_vectors = torch.stack(projected_vectors, dim=0)  # (N, top_r*top_r)

    # 5. Greedy selection
    selected_indices = []
    remaining_indices = set(range(N))
    selected_vectors = []



    #total_energy = projected_vectors.norm(dim=-1).pow(2).sum().item()
    current_energy = 0.0

    while current_energy / total_energy < energy_threshold:
        best_idx = -1
        best_gain = -float('inf')

        for idx in remaining_indices:
            vec = projected_vectors[idx]  # (top_r*top_r,)
            if selected_vectors:
                # Project onto orthogonal complement
                selected_mat = torch.stack(selected_vectors, dim=0)  # (num_selected, D)

                Q, _ = torch.linalg.qr(selected_mat.T, mode='reduced')  # (D, k)
                projection = (Q @ (Q.T @ vec))  # (D,)

                #projection = (vec @ selected_mat.T) @ selected_mat  # (D,)
                vec_residual = vec - projection
            else:
                vec_residual = vec

            gain = vec_residual.norm().item()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        selected_vectors.append(projected_vectors[best_idx] / (projected_vectors[best_idx].norm() + 1e-8))
        current_energy += best_gain ** 2

    selected_indices = torch.tensor(selected_indices)

    print(f"Selected {len(selected_indices)} samples covering {current_energy / total_energy * 100:.2f}% of gradient space.")

    # 7. Optional plotting
    plt.figure(figsize=(8, 6))
    plt.plot(torch.arange(len(selected_indices))+1, (torch.tensor([current_energy / total_energy for _ in selected_indices])*100).numpy(), label='Cumulative Span Coverage')
    plt.xlabel('Number of Samples Selected')
    plt.ylabel('Coverage (%)')
    plt.title('Greedy Span Selection Coverage')
    plt.grid(True)
    plt.legend()
    plt.savefig('greedy_span_coverage.png', dpi=300)

    return selected_indices

def select_probe_greedy_span_unified_normalized_high_precision(
    cur_matrixs_list,
    energy_threshold=0.95,
    top_r=None
):
    """
    Greedy span selection across multiple attention blocks, with per-block normalization.
    Dynamically select samples that together span a certain percentage of gradient space.

    Args:
        cur_matrixs_list (List[torch.Tensor]): list of (Num, 768, 768) tensors for each block.
        energy_threshold (float): fraction of gradient space to cover.
        top_r (int, optional): number of top singular vectors to use.
        feature_mode (str): "trace" (default) or "flatten". How to extract projected features.

    Returns:
        selected_indices (torch.Tensor)
    """

    N = cur_matrixs_list[0].shape[0]

    # 1. Normalize each block independently
    normalized_cur_matrices = []
    for matrices in cur_matrixs_list:
        frob_norms = torch.norm(matrices.view(N, -1), dim=-1, p=2).view(N, 1, 1)  # (N, 1, 1)
        matrices_normalized = matrices / (frob_norms + 1e-8)
        normalized_cur_matrices.append(matrices_normalized)
    
    # 2. Compute global covariance C
    C_global = sum([matrices.sum(dim=0) for matrices in normalized_cur_matrices])  # (768, 768)

    # 3. SVD on global C
    U, _, _ = torch.linalg.svd(C_global)
    if top_r is not None:
        U = U[:, :top_r]  # (768, top_r)

    # 4. Compute projected sample vectors
    projected_vectors = []
    for i in range(N):
        x_cov_sum = sum([matrices[i] for matrices in normalized_cur_matrices])  # (768, 768)
        proj = U.T @ x_cov_sum @ U  # (top_r, top_r)
        proj_feat = proj.flatten()  # (top_r*top_r,)
        projected_vectors.append(proj_feat)
    projected_vectors = torch.stack(projected_vectors, dim=0)  # (N, D)

    # 5. Greedy selection with orthogonal residual updates
    selected_indices = []
    remaining_indices = set(range(N))

    residual_vectors = projected_vectors.clone()  # (N, D)
    selected_vectors = []

    total_energy = projected_vectors.norm(dim=-1).pow(2).sum().item()
    current_energy = 0.0

    # 假设 selected_vectors 已经初始化
    while current_energy / total_energy < energy_threshold:
        assert remaining_indices
        best_idx = -1
        best_gain = -float('inf')

        for idx in remaining_indices:
            vec = residual_vectors[idx]
            if selected_vectors:
                # 计算当前向量与已选向量的正交残差
                selected_mat = torch.stack(selected_vectors, dim=0)  # (num_selected, D)
                Q, _ = torch.linalg.qr(selected_mat.T, mode='reduced')  # (D, k)
                vec_residual = vec - (Q @ (Q.T @ vec)) 
            else:
                vec_residual = vec

            # 当前样本的能量
            gain = vec_residual.norm().item() ** 2
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if not GREEDY:
                    break

        # 累计选择的样本能量
        selected_indices.append(best_idx)
        selected_vec = residual_vectors[best_idx] / (residual_vectors[best_idx].norm() + 1e-8)
        current_energy += projected_vectors[best_idx].norm().item() ** 2
        print(current_energy, '/', total_energy)

        # 更新 selected_vectors 和 residual_vectors
        selected_vectors.append(selected_vec)
        projection = (residual_vectors @ selected_vec.unsqueeze(-1)).squeeze(-1)
        residual_vectors = residual_vectors - projection.unsqueeze(-1) * selected_vec.unsqueeze(0)

        remaining_indices.discard(best_idx)

    # 输出最终的选择结果
    selected_indices = torch.tensor(selected_indices)
    print(f"Selected {len(selected_indices)} samples covering {current_energy / total_energy * 100:.2f}% of gradient space.")

    return selected_indices

# ------

class TopK:

    '''
    A class to maintain a collection of the top K items based on a specified attribute.

    This class allows for the dynamic addition of items, each represented as a dictionary, 
    where each dictionary must have a key 'proj_norm' that represents the value used 
    to determine the ranking. The class keeps track of the top K items with the highest 
    'proj_norm' values.
    '''

    def __init__(self, k):
        self.k = k
        self.top_k_list = []

    def add(self, dict):
        if len(self.top_k_list) < self.k:
            self.top_k_list.append(dict)
        elif dict['proj_norm'] > min(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm']:
            self.top_k_list.remove(min(self.top_k_list, key=lambda x: x['proj_norm']))
            self.top_k_list.append(dict)
        elif dict['proj_norm'] == min(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm'] and \
            dict['proj_norm'] == max(self.top_k_list, key=lambda x: x['proj_norm'])['proj_norm']:
            self.top_k_list.remove(min(self.top_k_list, key=lambda x: x['task_id']))
            self.top_k_list.append(dict)

    def get_top_k(self):
        return self.top_k_list

class SiNet(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]

        _set_random(os.environ["PYTHONHASHSEED"])
        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)])

        for name, module in self.backbone.named_modules():
            if 'transformer' in name and 'blocks' not in name:
                self.transformer_module = module

    def update_fc(self):
        self._cur_task_id += 1

    def forward(self, x, expert_id, inference = False):
        logits = []
        features = self.backbone(x, expert_id = expert_id)

        if inference:

            # Bayesian
            for i, prompts in enumerate(self.classifier_pool[:self._cur_task_id + 1]):
                # No Masking
                logits.append(prompts(features))

            logits = torch.cat(logits, dim=1)

            return logits

        else:
            logits.append(self.classifier_pool[self._cur_task_id](features))
            return torch.cat(logits, dim=1)

    def update_input_matrix(self, x):
        self.backbone(x, expert_id = -1, get_input_matrix = True)

class MInfLoRA(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]
        self.embd_dim = kwargs["embd_dim"]
        self.eval_mat = False

        self._known_classes = 0
        self.feature_list = []
        self.project_type = []

        self.distributed = torch.distributed.is_initialized()
        assert not self.distributed, 'current not support'
        self.local_rank = torch.distributed.get_rank() if self.distributed else 0
        
        self._network = SiNet(backbone, **kwargs)

        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_MaskedLoRA1)]

        # TRGP Implementation
        self.feature_list_each_tasks = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]
        self.final_decision = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]
        self.before_mat = [[0 for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]

        self.experts_distributions = []

        # Class Alignment Implementation
        self._use_class_alignment = kwargs['use_ca']
        self._class_means = None
        self._class_covs = None
        self._dataset = kwargs['dataset']
        if self._dataset == 'cifar':
            self.logit_norm = None
        else:
            self.logit_norm = 0.1   
    
        self.lll = []

        self._network.to(self.device)
        
    def observe(self, data):

        with torch.no_grad():
            self._network(self.probe_selection, expert_id = -1)

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(x, expert_id = self._network._cur_task_id)
        loss = F.cross_entropy(logits, y)

        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc, loss
    
    def inference(self, data, **kwargs):

        task_id = kwargs['task_id'] if 'task_id' in kwargs else -1
        x, y = data['image'].to(self.device, non_blocking=True), data['label'].to(self.device, non_blocking=True)

        logits = self._network(x, expert_id = task_id, inference = True)
        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc

    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        print('Greedy', GREEDY) # current best is not greedy, yes approx feature
        print('Approx Feature', APPROX_FEAT)
        
        self._network.update_fc()

       # mag = nn.ParameterList([nn.Parameter(torch.Tensor([1.0])) for _ in range(task_idx + 1)])
        _set_random(os.environ["PYTHONHASHSEED"])
        for module in self.attention_modules:
            #module.mag_lora = mag
            module.init_param()

        self._network = self._network.to(self.device)
        self._update_input_matrix(train_loader)
        
        '''
        probe_indices_svd = select_probe_svd_energy_matrix_unified_normalized(
            [m.cur_matrixs for m in self.attention_modules]
            ,probe_size=512
        )
        '''
        '''
        self.probe_indices_svd = select_probe_svd_energy_matrix_unified_normalized(
            [m.cur_matrixs for m in self.attention_modules]
            ,energy_threshold=0.15, top_r=64
        )
        '''
        self.probe_indices_svd = select_probe_greedy_span_unified_normalized_high_precision(
            [m.cur_matrixs for m in self.attention_modules]
            ,energy_threshold=0.01, top_r=128
            #,energy_threshold=0.5, top_r=128
        )
        
        self.probe_selection = self.dataset[self.probe_indices_svd].to(self.device)

        if task_idx == 0:
            for i, module in enumerate(self.attention_modules):

                # Either divide with 512 or divice with 512 * 197
                U, _, _ = torch.linalg.svd(module.cur_matrixs[self.probe_indices_svd].sum(dim=0) / 512, full_matrices=False)
                
                module.lora_A_k_list[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v_list[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))

        else:
            for i, module in enumerate(self.attention_modules):

                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T)
                module.feature_mat = feature_mat.clone().to(self.device)

                activation = module.cur_matrixs[self.probe_indices_svd].sum(dim=0) / 512
                activation = activation - feature_mat @ activation

                U, _, _ = torch.linalg.svd(activation, full_matrices = False)

                module.lora_A_k_list[task_idx].weight.data.copy_(U[:, :module.lora_rank].T/(3 ** 0.5))
                module.lora_A_v_list[task_idx].weight.data.copy_(U[:, :module.lora_rank].T/(3 ** 0.5))
    
        '''
        for i, module in enumerate(self.attention_modules):
            
            topk = TopK(1)

            mat = module.cur_matrix.cpu().numpy()
            mat_norm = np.linalg.norm(mat)

            for task_id in range(task_idx):

                if not np.array_equal(self.feature_list_each_tasks[task_id][i], np.zeros((1))):
            
                    proj_norm = np.linalg.norm(self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T @ mat)
                    print(f'{task_idx} to {task_id} in layer {i} : {proj_norm}')
                    
                    if proj_norm > Epsilon * mat_norm:
                        topk.add({'proj_norm':proj_norm, 'task_id': task_id})

            self.final_decision[task_idx][i] = [dic['task_id'] for dic in topk.get_top_k()]
            print(f'Layer {i} of {task_idx} consider {self.final_decision[task_idx][i]} as trust region')

        self.prev_matrix = []
        if task_idx == 0:
            for i, module in enumerate(self.attention_modules):
                
                U, _, _ = torch.linalg.svd(module.cur_matrix)
                U = torch.Tensor(U).to(self.device)

                self.prev_matrix.append(U[:,:module.lora_rank].T.cpu())

                module.lora_A_k_list[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v_list[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                #module.reset_input_matrix()
        else:
            for i, module in enumerate(self.attention_modules):
                assert self.project_type[i] == 'remove' or self.project_type[i] == 'retain'

                cur_matrix = module.cur_matrix.to(self.device)


                # TRGP
                tr = self.final_decision[task_idx][i][0]
                tr = task_idx - 1

                #feature_mat = torch.Tensor(self.feature_list_each_tasks[tr][i] @ self.feature_list_each_tasks[tr][i].T).to(self.device)

                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T).to(self.device)
                intersect = feature_mat @ cur_matrix

                target_shape = 768

                U, _, _ = np.linalg.svd(intersect.cpu().numpy(), full_matrices = False)
                U = torch.Tensor(U).to(self.device)
                module.space_k[tr] = U[:, :target_shape].T/math.sqrt(3)
                module.space_v[tr] = U[:, :target_shape].T/math.sqrt(3)

                # InfLoRA
                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T).to(self.device)

                if self.project_type[i] == 'remove':
                    cur_matrix = cur_matrix - feature_mat @ cur_matrix
                else:
                    cur_matrix = feature_mat @ cur_matrix

                module.feature_mat = feature_mat.clone()

                U, _, _ = np.linalg.svd(cur_matrix.cpu().numpy(), full_matrices = False)
                U = U[:, :module.lora_rank]

                alphas = torch.linalg.lstsq(torch.Tensor(module.lora_A_k_list[task_idx-1].weight.data).T.cpu(), torch.Tensor(U) / math.sqrt(3))
                if alphas.residuals.numel() != 0:
                    print(f'Task {task_idx}, Layer {i}, {alphas.residuals}')
                    assert 0

                U = torch.Tensor(U).to(self.device)

                module.lora_A_k_list[task_idx].weight.data.copy_(U[:, :module.lora_rank].T/math.sqrt(3)) # here should have /sqrt3
                module.lora_A_v_list[task_idx].weight.data.copy_(U[:, :module.lora_rank].T/math.sqrt(3))
        '''

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if f"classifier_pool.{task_idx}" in name or \
               f"lora_B_k_list.{task_idx}" in name or \
               f"lora_B_v_list.{task_idx}" in name:
                param.requires_grad_(True)

        for name, param in self._network.named_parameters():
            if param.requires_grad:
                print(name)

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

        self._update_feature(task_idx, train_loader, test_loaders)

    @torch.no_grad()
    def _update_feature(self, task_idx, train_loader, test_loaders):
        '''
        Update feature lists and the corresponding type
        '''

        self._update_input_matrix(train_loader)

        if self.local_rank == 0:

            threshold = (self.lame - self.lamb)*task_idx/self.task_num + self.lamb

            if task_idx == 0:
                for i, module in enumerate(self.attention_modules):
                    
                    activation = module.cur_matrixs[self.probe_indices_svd].sum(dim=0) / 512
                    U, S, _ = torch.linalg.svd(activation, full_matrices=False)
                    true_U = U[:, :module.lora_rank]

                    # Least Square
                    alphas = torch.linalg.lstsq(module.lora_A_k_list[task_idx].weight.data.T.cpu() * math.sqrt(3), true_U)
                    approx2_U = module.lora_A_k_list[task_idx].weight.data.T.cpu() * math.sqrt(3) @ alphas.solution

                    if APPROX_FEAT:
                        self.feature_list.append(approx2_U)
                        self.feature_list_each_tasks[task_idx][i] = approx2_U
                    else:
                        self.feature_list.append(true_U)
                        self.feature_list_each_tasks[task_idx][i] = true_U

                    self.project_type.append('remove')

            else:
                for i, module in enumerate(self.attention_modules):

                    activation = module.cur_matrixs[self.probe_indices_svd].sum(dim=0) / 512
                    act_hat = activation - torch.Tensor(self.feature_list[i] @ self.feature_list[i].T) @ activation

                    U, _, _ = torch.linalg.svd(act_hat, full_matrices = False)
                    true_U = U[:, :module.lora_rank]

                    alphas = torch.linalg.lstsq(module.lora_A_k_list[task_idx].weight.data.T.cpu() * math.sqrt(3), true_U)
                    approx2_U = module.lora_A_k_list[task_idx].weight.data.T.cpu() * math.sqrt(3) @ alphas.solution

                    if APPROX_FEAT:
                        self.feature_list[i] = torch.cat([self.feature_list[i], approx2_U], dim=1)
                        self.feature_list_each_tasks[task_idx][i] = approx2_U
                    else:
                        self.feature_list[i] = torch.cat([self.feature_list[i], true_U], dim=1)
                        self.feature_list_each_tasks[task_idx][i] = true_U

            print('-'*40)
            print(f'Threshold: {threshold}')
            print('-'*40)
            for i in range(len(self.feature_list)):
                '''
                if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                    feature = self.feature_list[i]
                    U, S, V = np.linalg.svd(feature)
                    new_feature = U[:,feature.shape[1]:]
                    self.feature_list[i] = new_feature
                    self.project_type[i] = 'retain'
                elif self.project_type[i]=='retain':
                    assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
                '''
                print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
            print('-'*40)

    @torch.no_grad()
    def _update_input_matrix(self, train_loader):

        for module in self.attention_modules:
            module.reset_input_matrix()

        _set_random(os.environ["PYTHONHASHSEED"]) # consistency
        self.dataset = []
        for batch in tqdm(train_loader, desc="Forwarding to get input matrix", disable=(self.local_rank != 0)):
            self._network.update_input_matrix(batch['image'].to(self.device))
            self.dataset.append(batch['image'])

        self.dataset = torch.cat(self.dataset, dim=0)

        for module in self.attention_modules:
            module.cur_matrixs = torch.cat(module.cur_matrixs, dim=0)
            module.cur_matrixs = torch.bmm(
                module.cur_matrixs.permute(0, 2, 1),
                module.cur_matrixs
            ).cpu()

    def get_parameters(self, config):
        return self._network.parameters()        