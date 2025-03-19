"""
@article{lin2022trgp,
  title={TRGP: Trust Region Gradient Projection for Continual Learning},
  author={Lin, Sen and Yang, Li and Fan, Deliang and Zhang, Junshan},
  journal={arXiv preprint arXiv:2202.02931},
  year={2022}
}

Code Reference:
https://github.com/LYang-666/TRGP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbone.petl.adapter import MaskedAdapter
from .backbone.clip import tokenize, CLIP

Epsilon = 0.5

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

    def get_top_k(self):
        return self.top_k_list

class TRGP_CLIP(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.network = backbone
        self.device = device

        self.task_num = kwargs['task_num']
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.label_smoothing = kwargs['label_smoothing']

        self._known_classes = 0
        self.visual_U = []
        self.lamda = [[0 for _ in range(12)] for _ in range(12)]
        self.lamda_scale = kwargs['lamda_scale']

        self.accm_class_names = []   
        self.curr_class_names = []
        self.accm_text_tokens = None
        self.curr_text_tokens = None

        self.prompt_template = kwargs['prompt_template']
        
        self.feature_list = []
        self.feature_mat = []

        self.layers = []
        for name, module in self.network.named_modules():
            if 'visual' in name and isinstance(module, MaskedAdapter):
                self.layers.append(module)

        self.down_proj = [[0 for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.up_proj = [[0 for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.feature_list_each_tasks = [[np.zeros((1)) for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.scale_param_each_tasks_each_layers = [[np.zeros((1)) for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.all_space = [[np.zeros((1)) for _ in range(len(self.layers))] for _ in range(self.task_num)]

        for name, param in self.network.named_parameters():
            param.requires_grad = False
            if 'adaptmlp' in name:
                param.requires_grad = True

        self.network.to(self.device)

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.curr_text_tokens)
        loss = F.cross_entropy(logits_per_img, y, label_smoothing=self.label_smoothing)

        preds = logits_per_img.softmax(dim=-1).argmax(dim=1)

        loss.backward()
        
        if self.cur_task > 0:
            for i, module in enumerate(self.layers):
                sz = module.scale_proj.weight.grad.data.shape[0]
                module.scale_proj.weight.grad.data = module.scale_proj.weight.grad.data - (module.scale_proj.weight.grad.data.view(sz,-1) @ self.feature_mat[i].to(self.device)).view(module.scale_proj.weight.shape)

        acc = preds.eq(y).sum().item() / y.size(0)

        return preds, acc, loss
    
    def inference(self, data, task_id=-1):
        
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        # Task-Aware (Task-Incremetanl Scenario)
        if task_id > -1:

            for i, module in enumerate(self.layers):

                module.down_proj = self.down_proj[task_id][i]
                module.space = self.all_space[task_id][i]
                module.scale_param = nn.ParameterList([
                    nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[task_id][i]
                ])
                module.up_proj = self.up_proj[task_id][i]

            features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.accm_text_tokens[task_id * self.inc_cls_num : (task_id + 1) * self.inc_cls_num])

            preds = logits_per_img.softmax(dim=-1).argmax(dim=1) + task_id * self.inc_cls_num

        # Task-Agnostic (Class-Incremetanl Scenario)
        else:
            
            logits = []
            for t in range(self.cur_task + 1):
                
                for i, module in enumerate(self.layers):

                    module.down_proj = self.down_proj[t][i]
                    module.space = self.all_space[t][i]
                    module.scale_param = nn.ParameterList([
                        nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[t][i]
                    ])
                    module.up_proj = self.up_proj[t][i]

                features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.accm_text_tokens[t * self.inc_cls_num : (t + 1) * self.inc_cls_num])
                logits.append(logits_per_img)

            preds = torch.cat(logits, dim=-1).softmax(dim=-1).argmax(dim=1)

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        # Last task have leave scale_param and space, need to init again
        for module in self.layers:
            module.disable_scale()

        self.cur_task = task_idx

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        self.curr_class_names = train_loader.dataset.get_class_names()
        self.accm_class_names += self.curr_class_names

        self.curr_text_tokens = tokenize(
            [self.prompt_template.format(c) for c in self.curr_class_names]
        ).to(self.device)

        self.accm_text_tokens = tokenize(
            [self.prompt_template.format(c) for c in self.accm_class_names]
        ).to(self.device)


        if task_idx > 0:

            # temp compute and save them for training later
            self.feature_mat = [torch.tensor(feat @ feat.T, dtype=torch.float32) for feat in self.feature_list] 

            optimizer = torch.optim.SGD(self.network.parameters(), lr = 0.01) # lr hardcoded

            x, y = [], []
            for batch in train_loader:
                x.append(batch['image'].to(self.device))
                y.append(batch['label'].to(self.device) - self._known_classes)

            x, y = torch.cat(x, dim = 0), torch.cat(y, dim = 0)

            indices = torch.randperm(x.size(0))
            selected_indices = indices[:125]
            x, y = x[selected_indices], y[selected_indices]
            optimizer.zero_grad()  
            features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.curr_text_tokens)
            loss = F.cross_entropy(logits_per_img, y)
            loss.backward()





            for i, module in enumerate(self.layers):

                topk = TopK(2)

                grad = module.scale_proj.weight.grad.data.detach().cpu().numpy()

                print(grad.shape)
                print(self.feature_list_each_tasks[0][i].shape)

                for task_id in range(task_idx):
                    
                    # Projection of down_proj grad into feature spaces of old data
                    proj = grad @ self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T
                    proj_norm = np.linalg.norm(proj)

                    print(f'Down Proj of Layer {i} of {task_idx} to {task_id} : {proj_norm:.4f}/{np.linalg.norm(grad):.4f} ({proj_norm > Epsilon * np.linalg.norm(grad)})')

                    if proj_norm > Epsilon * np.linalg.norm(grad):
                        topk.add({'proj_norm':proj_norm, 'task_id': task_id})

        
                final_decision = [dic['task_id'] for dic in topk.get_top_k()]

                module.enable_scale(
                    [torch.tensor(self.feature_list_each_tasks[task_id][i], dtype=torch.float32).to(self.device) for task_id in final_decision]
                )

                print(f'Proj of Layer {i} of {task_idx} consider {final_decision} as trust region')

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        # Save the scale param
        for i, module in enumerate(self.layers):
            self.down_proj[task_idx][i] = module.down_proj
            self.up_proj[task_idx][i] = module.up_proj
            
            self.scale_param_each_tasks_each_layers[task_idx][i] = [scale_param.data for scale_param in module.scale_param] # top2
            self.all_space[task_idx][i] = module.space
            module.disable_scale()


        x = []
        for batch in train_loader:
            x.append(batch['image'].to(self.device))
        x = torch.cat(x, dim = 0)

        # hardcoded, choose 125 input from it
        indices = torch.randperm(x.size(0))
        selected_indices = indices[:125]
        x = x[selected_indices]

        self.network.eval()
        self.network(x, self.curr_text_tokens, compute_input_matrix = True)

        mat_list = []
        for module in self.layers:

            assert module.input_matrix.shape[0] == 125
            input_matrix = module.input_matrix.view(-1, module.input_matrix.shape[-1]).detach().cpu().numpy().T
            mat_list.append(input_matrix)

        threshold = 0.97 + task_idx * 0.003

        # get the space for each layer
        if task_idx == 0:

            for i, activation in enumerate(mat_list):

                U, S, _ = np.linalg.svd(activation, full_matrices = False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)

                self.feature_list_each_tasks[task_idx][i] = U[:, :r]
                self.feature_list.append(U[:, :r]) # space_list_all in trgp original code

        else:

            for i, activation in enumerate(mat_list):

                _, S, _ = np.linalg.svd(activation, full_matrices = False)
                sval_total = (S**2).sum()

                delta = (self.feature_list[i].T @ activation @ activation.T @ self.feature_list[i]).diagonal()

                # following the GPM to get the sigma (S**2)
                act_hat = activation - self.feature_list[i] @ self.feature_list[i].T @ activation
                U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
                sigma = S**2

                # stack delta and sigma, then sort in descending order
                stack = np.hstack((delta, sigma))
                stack_index = np.argsort(stack)[::-1] # the index of each element in descending sorted array
                stack = np.sort(stack)[::-1] # descending sorted array

                if threshold * sval_total <= 0:
                    r = 0
                else:
                    r = min(np.sum(np.cumsum(stack) < threshold * sval_total) + 1, activation.shape[0])

                Ui = np.hstack((self.feature_list[i], U))
                sel_each = stack_index[:r]
                sel_overall = sel_each[sel_each >= len(delta)] # without overlap

                self.feature_list[i] = np.hstack((self.feature_list[i], Ui[:, sel_overall]))
                self.feature_list_each_tasks[task_idx][i] = Ui[:, sel_each]

                if sel_overall.shape[0] == 0:
                    print(f'Skip Updating Space for layer: {i+1}')

    def get_parameters(self, config):
        return self.network.parameters()