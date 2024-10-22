import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from collections.abc import Iterable

from .backbone.alexnet_trgp import Conv2d, Linear

Epsilon_1 = 0.5

class Network(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone

        self.classifiers = nn.ModuleList([])
        self.classifiers.append(
            nn.Linear(2048, kwargs['init_cls_num'], bias = False)
        )

        for _ in range(kwargs['task_num'] - 1):
            self.classifiers.append(
                nn.Linear(2048, kwargs['inc_cls_num'], bias = False)
            )


    def return_hidden(self, data):
        return self.backbone(data)
    
    def forward(self, data, compute_input_matrix = False):
        logits = []
        image_features = self.backbone(data, compute_input_matrix)
        for classifier in self.classifiers:
            logits.append(classifier(image_features))

        return logits
        #return torch.cat(logits, dim=1)

class TRGP(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.network = Network(backbone, **kwargs)
        self.device = device

        self.task_num = kwargs["task_num"]
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.number_of_layer = 5 # hardcoded for alexnet
        self._known_classes = 0

        self.feature_list = []
        self.feature_list_each_tasks = [[np.zeros((1)) for _ in range(self.number_of_layer)] for _ in range(self.task_num)]
        self.scale_param_each_tasks_each_layers = [[np.zeros((1)) for _ in range(self.number_of_layer)] for _ in range(self.task_num)]
        self.all_space = [[np.zeros((1)) for _ in range(self.number_of_layer)] for _ in range(self.task_num)]

        self.feature_mat = []

        self.layers = [] # 3 Conv2d, Then 2 Linear
        for module in self.network.modules():
            if isinstance(module, Conv2d):
                self.layers.append(module)
            if isinstance(module, Linear):
                self.layers.append(module)

        self.network.to(self.device)

    def observe(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self.network(x)
        loss = F.cross_entropy(logits[self.cur_task], y)

        preds = logits[self.cur_task].max(1)[1]
        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        loss.backward()
        
        if self.cur_task > 0:
            kk = 0 
            for name, param in self.network.named_parameters():
                if ('fc' in name or 'conv' in name) and 'weight' in name:
                    sz =  param.grad.data.size(0)
                    param.grad.data = param.grad.data - (param.grad.data.view(sz,-1) @ self.feature_mat[kk]).view(param.size())
                    kk +=1

        return preds, acc, loss
    
    def inference(self, data, task_id):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        for i, module in enumerate(self.layers):
            module.scale_param = nn.ParameterList([
                nn.Parameter(scale_param) for scale_param in self.scale_param_each_tasks_each_layers[task_id][i]
            ])
            module.space = self.all_space[task_id][i]

        logits = self.network(x)

        preds = logits[task_id].max(1)[1]
        preds += task_id * 10

        correct_count = preds.eq(y).sum().item()
        acc = correct_count / y.size(0)

        return preds, acc

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self.cur_task = task_idx

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num

        if task_idx > 0:

            # temp compute and save them for training later
            self.feature_mat = [torch.Tensor(feat @ feat.T).to(self.device) for feat in self.feature_list] 

            optimizer = torch.optim.SGD(self.network.parameters(), lr = 0.0005) # lr hardcoded

            x, y = None, None
            for batch_idx, batch in enumerate(train_loader):

                image, label = batch['image'].to(self.device), batch['label'].to(self.device) - self._known_classes

                if x is not None:
                    x = torch.cat((x, image), dim=0)
                else:
                    x = image

                if y is not None:
                    y = torch.cat((y, label), dim=0)
                else:
                    y = label

            indices = torch.randperm(x.size(0))
            selected_indices = indices[:125]
            x, y = x[selected_indices], y[selected_indices]
            optimizer.zero_grad()  
            logits = self.network(x)
            loss = F.cross_entropy(logits[self.cur_task], y)
            loss.backward()  

            grad_list = []

            for name, param in self.network.named_parameters():
                if 'conv' in name and 'weight' in name: # weight of conv
                    grad = param.grad.data.detach().cpu().numpy()
                    grad = grad.reshape(grad.shape[0], grad.shape[1] * grad.shape[2] * grad.shape[3])
                    grad_list.append(grad)
                if 'fc' in name and 'weight' in name: # weight of linear
                    grad = param.grad.data.detach().cpu().numpy()
                    grad_list.append(grad)
            
            class TopK:
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

            for i, module in enumerate(self.layers):

                topk = TopK(2)

                for task_id in range(task_idx):

                    grad = grad_list[i]
                    proj = grad @ self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T
                    proj_norm = np.linalg.norm(proj)

                    print(f'Layer {i} of {task_idx} to {task_id} : {proj_norm:.4f}/{np.linalg.norm(grad):.4f} ({proj_norm > Epsilon_1 * np.linalg.norm(grad)})')

                    if proj_norm > Epsilon_1 * np.linalg.norm(grad):
                        topk.add({'proj_norm':proj_norm, 'task_id': task_id})

                final_decision = [dic['task_id'] for dic in topk.get_top_k()]
                module.enable_scale([
                    torch.tensor(self.feature_list_each_tasks[task_id][i], dtype=torch.float32).to(self.device) for task_id in final_decision
                ])
                print(f'Layer {i} of {task_idx} consider {final_decision} as trust region')

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

    # Save the scale param
        for i, module in enumerate(self.layers):
            print(f'layer {i} of task {task_idx} has scale {[scale_param.data for scale_param in module.scale_param]}')
            self.scale_param_each_tasks_each_layers[task_idx][i] = [scale_param.data for scale_param in module.scale_param] # top2
            self.all_space[task_idx][i] = module.space # top2
            module.disable_scale()

        mat_list = [] # representation (activation) of each layer

        x = None

        for batch_idx, batch in enumerate(train_loader):

            if x is not None:
                x = torch.cat((x, batch['image'].to(self.device)), dim=0)
            else:
                x = batch['image'].to(self.device)

        # hardcoded, choose 125 input from it
        num_inputs = x.size(0)
        indices = torch.randperm(num_inputs)
        selected_indices = indices[:125]
        x = x[selected_indices]

        self.network.eval()
        self.network(x, compute_input_matrix = True)

        batch_list = [2*12,100,100,125,125] 
        map_list = [32, 14, 6, 1024, 2048] # harcoded, this is the input size of data in each layer of network
        ksize = [4, 3, 2] # kernel size of each conv layer
        conv_output_size = [29, 12, 5] # output size of each conv layer
        in_channel = [3, 64, 128] # input channel of each conv layer

        for i, module in enumerate(self.layers):
            k=0
            if isinstance(module, Conv2d):
                bsz = batch_list[i]
                ksz = ksize[i]
                s = conv_output_size[i]
                inc = in_channel[i]

                # act is the input of each layer (both conv and linear)

                mat = np.zeros((ksz * ksz * inc, s * s * bsz))
                act = module.input_matrix.detach().cpu().numpy()

                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:,k]=act[kk, :, ii:ksz+ii, jj:ksz+jj].reshape(-1) 
                            k += 1

                mat_list.append(mat)
            elif isinstance(module, Linear):
                mat_list.append(module.input_matrix.detach().cpu().numpy().T)

        threshold = 0.97 + task_idx * 0.003

        # get the space for each layer
        if task_idx == 0:

            for i in range(self.number_of_layer):

                # same with dualgpm
                activation = mat_list[i]
                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)

                self.feature_list_each_tasks[task_idx][i] = U[:, :r]
                self.feature_list.append(U[:, :r]) # space_list_all in trgp original code
        else:

            for i in range(self.number_of_layer):

                activation = mat_list[i]
                _, S, _=np.linalg.svd(activation, full_matrices = False)
                sval_total = (S**2).sum()

                # compute the projection using previous space (feature_list)
                
                R2 = activation @ activation.T
                delta = []

                for k in range(self.feature_list[i].shape[1]):
                    space = self.feature_list[i][:, k] # each column
                    delta.append(space.T @ R2 @ space)
                delta = np.array(delta) # (self.feature_list[i].shape[1] ,1)

                # following the GPM to get the sigma (S**2)

                act_hat = activation - self.feature_list[i] @ self.feature_list[i].T @ activation
                U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
                sigma = S**2

                # stack delta and sigma in a same list, then sort in descending order

                stack = np.hstack((delta, sigma))  #[0,..30, 31..99]
                stack_index = np.argsort(stack)[::-1]   #[99, 0, 4,7...]
                stack = np.sort(stack)[::-1]

                if threshold * sval_total <= 0:
                    r = 0
                else:
                    r = min(np.sum(np.cumsum(stack) < threshold * sval_total) + 1, activation.shape[0])

                #=5 save the corresponding space
                Ui = np.hstack((self.feature_list[i], U))

                self.feature_list_each_tasks[task_idx][i] = Ui[:, stack_index < r]

                # calculate how many space from current new task
                sel_index_from_U = stack_index[len(delta):] < r
                if np.any(sel_index_from_U):
                    # update the overall space without overlap
                    self.feature_list[i] = np.hstack((self.feature_list[i], U[:, sel_index_from_U] ))

    def get_parameters(self, config):
        return self.network.parameters()