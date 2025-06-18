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

from .backbone.alexnet import Conv2d_TRGP, Linear_TRGP, AlexNet_TRGP
from .backbone.clip import tokenize, CLIP

Epsilon = 0.5

AlexNet = AlexNet_TRGP
Clip = CLIP

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

class Network(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone

        self.classifiers = nn.ModuleList([
            nn.Linear(backbone.feat_dim, kwargs['init_cls_num'], bias = False)] + 
            [nn.Linear(backbone.feat_dim, kwargs['inc_cls_num'], bias = False) for _ in range(kwargs['task_num'] - 1)]
        )

    def return_hidden(self, data):
        return self.backbone(data)
    
    def forward(self, data, compute_input_matrix = False):
        logits = []
        image_features = self.backbone(data, compute_input_matrix)
        for classifier in self.classifiers:
            logits.append(classifier(image_features))

        return logits

class TRGP(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.backbone = backbone
        self.device = device
        self.task_num = kwargs["task_num"]
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.label_smoothing = kwargs['label_smoothing']

        self._known_classes = 0
        self.feature_list = []
        self.feature_mat = []
        self.layers = [] 

        if isinstance(backbone, Clip):
            self.network = backbone

            self.visual_U = []
            self.lamda = [[0 for _ in range(12)] for _ in range(12)]

            self.accm_class_names = []   
            self.curr_class_names = []
            self.accm_text_tokens = None
            self.curr_text_tokens = None

            self.prompt_template = kwargs['prompt_template']

            # 12 Visual Transformer's Adapter * 2 (Down & Up)
            for name, module in self.network.named_modules():
                if 'visual' in name and isinstance(module, Linear_TRGP):
                    self.layers.append(module)
            
            for name, param in self.network.named_parameters():
                param.requires_grad = False
                if 'adaptmlp' in name:
                    param.requires_grad = True

        elif isinstance(backbone, AlexNet):
            self.network = Network(backbone, **kwargs)

            # # 3 Conv2d and 2 Linear
            for module in self.network.modules(): # 
                if isinstance(module, Conv2d_TRGP) or isinstance(module, Linear_TRGP):
                    self.layers.append(module)

        else:
            raise NotImplementedError

        self.feature_list_each_tasks = [[0 for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.scale_param_each_tasks_each_layers = [[0 for _ in range(len(self.layers))] for _ in range(self.task_num)]
        self.all_space = [[0 for _ in range(len(self.layers))] for _ in range(self.task_num)]

        self.network.to(self.device)

    def observe(self, data):
        
        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes
        if len(y) == 1: # Ignore batch_size == 1
            return None, 0., torch.zeros(1, requires_grad=True)

        if isinstance(self.backbone, Clip):

            features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.curr_text_tokens)
            loss = F.cross_entropy(logits_per_img, y, label_smoothing=self.label_smoothing)

            preds = logits_per_img.softmax(dim=-1).argmax(dim=1)

            loss.backward()
            
            if self.cur_task > 0:
                for i, module in enumerate(self.layers):
                    sz = module.weight.grad.data.shape[0]
                    module.weight.grad.data = module.weight.grad.data - (module.weight.grad.data.view(sz,-1) @ self.feature_mat[i]).view(module.weight.shape)

        elif isinstance(self.backbone, AlexNet):

            logits = self.network(x)
            loss = F.cross_entropy(logits[self.cur_task], y, label_smoothing=self.label_smoothing)

            preds = logits[self.cur_task].max(1)[1]
            
            loss.backward()
            
            if self.cur_task > 0:
                for i, module in enumerate(self.layers):
                    sz = module.weight.grad.data.shape[0]
                    module.weight.grad.data = module.weight.grad.data - (module.weight.grad.data.view(sz,-1) @ self.feature_mat[i]).view(module.weight.shape)

        else:
            raise NotImplementedError

        acc = preds.eq(y).sum().item() / y.size(0)

        return preds, acc, loss
    
    def inference(self, data, task_id = -1):

        x, y = data['image'].to(self.device), data['label'].to(self.device)
        
        # Add dummy, to prevert batch_size == 1
        dummy_x = torch.randn_like(x[:1])  # only one dummy sample
        x = torch.cat([x, dummy_x], dim=0)

        # Task-Aware (Task-Incremetanl Scenario)
        if task_id > -1:

            if task_id == 0:
                bias_classes = 0
            elif task_id == 1:
                bias_classes = self.init_cls_num
            else:
                bias_classes = self.init_cls_num + (task_id - 1) * self.inc_cls_num

            if isinstance(self.backbone, Clip):

                for i, module in enumerate(self.layers):
                    module.space = self.all_space[task_id][i]
                    module.scale_param = nn.ParameterList([nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[task_id][i]])

                features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.accm_text_tokens[bias_classes : self.init_cls_num + task_id * self.inc_cls_num])
                preds = logits_per_img.softmax(dim=-1).argmax(dim=1) + bias_classes

            elif isinstance(self.backbone, AlexNet):

                for i, module in enumerate(self.layers):
                    module.space = self.all_space[task_id][i]
                    module.scale_param = nn.ParameterList([nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[task_id][i]])

                logits = self.network(x)
                preds = logits[task_id].softmax(dim=-1).argmax(dim=1) + bias_classes

            else:
                raise NotImplementedError

        # Task-Agnostic (Class-Incremetanl Scenario)
        else:

            logits = []

            if isinstance(self.backbone, Clip):

                for t in range(self.cur_task + 1):
                    for i, module in enumerate(self.layers):
                        
                        module.space = self.all_space[t][i]
                        module.scale_param = nn.ParameterList([nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[t][i]])

                    if t == 0:
                        features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.accm_text_tokens[: self.init_cls_num])
                    else:
                        features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.accm_text_tokens[self.init_cls_num + (t-1) * self.inc_cls_num : self.init_cls_num + t * self.inc_cls_num])
                    
                    logits.append(logits_per_img)

            elif isinstance(self.backbone, AlexNet):

                for t in range(self.cur_task + 1):
                    for i, module in enumerate(self.layers):
                        module.space = self.all_space[t][i]
                        module.scale_param = nn.ParameterList([nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[t][i]])
                        
                    logits.append(self.network(x)[t])

            else:
                raise NotImplementedError

            preds = torch.cat(logits, dim=-1).softmax(dim=-1).argmax(dim=1)

        # Remove dummy
        preds = preds[:-1]

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        # Last task have scale_param and space, need to init again
        for module in self.layers:
            module.disable_scale()

        self.cur_task = task_idx

        if isinstance(self.backbone, Clip):
        
            self.curr_class_names = train_loader.dataset.get_class_names()
            self.accm_class_names += self.curr_class_names

            self.curr_text_tokens = tokenize([self.prompt_template.format(c) for c in self.curr_class_names]).to(self.device)
            self.accm_text_tokens = tokenize([self.prompt_template.format(c) for c in self.accm_class_names]).to(self.device)

        if task_idx > 0:

            self.feature_mat = [torch.tensor(feat @ feat.T, dtype=torch.float32, device=self.device) for feat in self.feature_list] 
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

            if isinstance(self.backbone, Clip):

                features_img, features_txt, logits_per_img, logits_per_txt = self.network(x, self.curr_text_tokens)
                loss = F.cross_entropy(logits_per_img, y)

            elif isinstance(self.backbone, AlexNet):

                logits = self.network(x)
                loss = F.cross_entropy(logits[self.cur_task], y)
            
            loss.backward()

            for i, module in enumerate(self.layers):

                topk = TopK(2)

                grad = module.weight.grad.data.detach().cpu().numpy()
                if isinstance(self.backbone, AlexNet) and isinstance(module, Conv2d_TRGP):
                    grad = grad.reshape(grad.shape[0], -1)

                for task_id in range(task_idx):

                    proj = grad @ self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T
                    proj_norm = np.linalg.norm(proj)

                    print(f'Layer {i} of {task_idx} to {task_id} : {proj_norm:.4f}/{np.linalg.norm(grad):.4f} ({proj_norm > Epsilon * np.linalg.norm(grad)})')

                    if proj_norm > Epsilon * np.linalg.norm(grad):
                        topk.add({'proj_norm':proj_norm, 'task_id': task_id})

                final_decision = [dic['task_id'] for dic in topk.get_top_k()]
                module.enable_scale([
                    torch.tensor(self.feature_list_each_tasks[task_id][i], dtype=torch.float32).to(self.device) for task_id in final_decision
                ])
                print(f'Layer {i} of {task_idx} consider {final_decision} as trust region')

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        self._known_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

        # Save the scale param
        for i, module in enumerate(self.layers):
            self.scale_param_each_tasks_each_layers[task_idx][i] = [scale_param.data for scale_param in module.scale_param] # top2
            self.all_space[task_idx][i] = module.space
            module.disable_scale()

        x = torch.cat([batch['image'].to(self.device) for batch in train_loader], dim = 0)

        # hardcoded, choose 125 input from it
        indices = torch.randperm(x.size(0))
        selected_indices = indices[:125]
        x = x[selected_indices]

        self.network.eval()

        mat_list = [] # Representation / Activation of each layer
        threshold = 0.97 + task_idx * 0.003

        if isinstance(self.backbone, Clip):

            self.network(x, self.curr_text_tokens, compute_input_matrix = True)

            for module in self.layers:

                assert module.input_matrix.shape[0] == 125
                mat_list.append(module.input_matrix.view(-1, module.input_matrix.shape[-1]).detach().cpu().numpy().T)

        elif isinstance(self.backbone, AlexNet):

            self.network(x, compute_input_matrix = True)
            
            batch_list = [2*12,100,100] 
            ksize = [4, 3, 2] # kernel size of each conv layer
            conv_output_size = [29, 12, 5] # output size of each conv layer
            in_channel = [3, 64, 128] # input channel of each conv layer

            for i, module in enumerate(self.layers):
                
                if isinstance(module, Conv2d_TRGP):
                    bsz, ksz, s, inc = batch_list[i], ksize[i], conv_output_size[i], in_channel[i]

                    mat = np.zeros((ksz * ksz * inc, s * s * bsz))
                    act = module.input_matrix.detach().cpu().numpy()

                    k = 0
                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                mat[:,k]=act[kk, :, ii:ksz+ii, jj:ksz+jj].reshape(-1) 
                                k += 1

                    mat_list.append(mat)
                elif isinstance(module, Linear_TRGP):
                    mat_list.append(module.input_matrix.detach().cpu().numpy().T)

        # get the space for each layer
        if task_idx == 0:

            for i, activation in enumerate(mat_list):

                U, S, _ = np.linalg.svd(activation, full_matrices = False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)

                self.feature_list_each_tasks[task_idx][i] = U[:, :r]
                self.feature_list.append(U[:, :r]) 
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