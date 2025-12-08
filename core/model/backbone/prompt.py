# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SmithKGCKAPFK23,
  author       = {James Seale Smith and
                  Leonid Karlinsky and
                  Vyshnavi Gutta and
                  Paola Cascante{-}Bonilla and
                  Donghyun Kim and
                  Assaf Arbelle and
                  Rameswar Panda and
                  Rog{\'{e}}rio Feris and
                  Zsolt Kira},
  title        = {CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free
                  Continual Learning},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2023, Vancouver, BC, Canada, June 17-24, 2023},
  pages        = {11909--11919},
  publisher    = {{IEEE}},
  year         = {2023}
}

https://arxiv.org/abs/2211.13218

Adapted from https://github.com/GT-RIPL/CODA-Prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import copy

# source code from https://github.com/GT-RIPL/CODA-Prompt
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full parameters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)
            

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            # print(p.shape)
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()       
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0
            

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(nn.Module):

    def __init__(self, length, prompt_init=nn.init.uniform_, prompt_key=False, 
                 pool_size=None, top_k=None, num_layers=1, embed_dim=768):

        super().__init__()
        self.length = length
        self.prompt_init = prompt_init
        self.pool_size = pool_size
        self.top_k = top_k
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Initialize prompt parameters
        self.prompt = nn.Parameter(
            torch.empty((self.num_layers, self.pool_size, self.length, embed_dim))
        )
        self.prompt_key = nn.Parameter(
            torch.empty((self.pool_size, embed_dim))
        )
        self.prompt_init(self.prompt)
        self.prompt_init(self.prompt_key)

    def forward(self, x_embed, cls_features=None):

        B, N, C = x_embed.shape
        assert C == self.embed_dim

        # Normalize key features
        prompt_key_norm = F.normalize(self.prompt_key, p=2, dim=-1, eps=1e-12)
        x_embed_norm = F.normalize(cls_features, p=2, dim=-1, eps=1e-12)

        sim = x_embed_norm @ prompt_key_norm.T
        _, idx = torch.topk(sim, self.top_k, dim=1)

        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        
        # Manually pad to pool_size, equivalent as jnp.unique()
        prompt_id = F.pad(prompt_id, (0, self.pool_size - len(prompt_id)), "constant", prompt_id[0])
        id_counts = F.pad(id_counts, (0, self.pool_size - len(id_counts)), "constant", 0)

        _, major_idx = torch.topk(id_counts, self.top_k)
        
        major_prompt_id = prompt_id[major_idx]
        idx = major_prompt_id.unsqueeze(0).repeat(B, 1)

        batched_prompt_raw = self.prompt[:, idx]

        batched_prompt = batched_prompt_raw.reshape(
            batched_prompt_raw.shape[0], 
            batched_prompt_raw.shape[1], 
            -1, 
            batched_prompt_raw.shape[-1]
        )
            
        # Calculate pull constraint loss
        batched_key_norm = prompt_key_norm[idx]
        sim_pull = batched_key_norm * x_embed_norm.unsqueeze(1)
        reduce_sim = torch.sum(sim_pull) / B

        return batched_prompt, reduce_sim

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    



# @inproceedings{10.24963/ijcai.2024/456,
#   author = {Hong, Chenxing and Jin, Yan and Kang, Zhiqi and Chen, Yizhou and Li, Mengke and Lu, Yang and Wang, Hanzi},
#   title = {Dynamically anchored prompting for task-imbalanced continual learning},
#   booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
#   year = {2025},
# }
class DAP(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',tasklength=10):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.tasklength = tasklength
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            generalpromt = (top_k, length, embed_dim)

            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                self.taskprompt = nn.ParameterList([nn.Parameter(torch.zeros(top_k, length, embed_dim)) for _ in range(tasklength)]) # this is for taskid
                self.generalprompt = nn.Parameter(torch.zeros(generalpromt))

            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                self.taskprompt = nn.ParameterList([nn.Parameter(torch.zeros(top_k, length, embed_dim)) for _ in range(tasklength)]) # this is for taskid
                for tp in self.taskprompt:
                    nn.init.uniform_(tp, -1, 1)
                self.generalprompt = nn.Parameter(torch.randn(generalpromt))
                nn.init.uniform_(self.generalprompt, -1, 1)

        if prompt_key:

            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:

            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None,taskid=None):

        out = dict()

        top_k, length, c = self.taskprompt[taskid].shape
        batched_task_prompt_raw = self.taskprompt[taskid].reshape(top_k * length, c)
        batched_task_prompt = batched_task_prompt_raw.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        batched_general_prompt_raw = self.generalprompt.reshape(top_k * length, c)
        batched_general_prompt = batched_general_prompt_raw.unsqueeze(0).expand(x_embed.shape[0], -1, -1)


        out['total_prompt_len'] = batched_task_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_task_prompt, x_embed], dim=1)

        out['gen_total_prompt_len'] = batched_general_prompt.shape[1]
        out['gen_prompted_embedding'] = torch.cat([batched_general_prompt, x_embed], dim=1)
        return out