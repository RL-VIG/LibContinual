import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
from math import pi
from torchvision import transforms
from .backbone.transformer import MultiHeadAttention_MoEMaskedLoRA

Epsilon = 0.5

class Model(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()

        self._cur_task_id = -1
        self.backbone = backbone

        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)])

    def update_fc(self):
        self._cur_task_id += 1

    def fc_only(self, x, expert_id):
        logits = []
        for prompts in self.classifier_pool[:expert_id + 1]:
            logits.append(prompts(x))
        return torch.cat(logits, dim=1)

    def get_feature(self, x, expert_id):
        features = self.backbone(x, expert_id = expert_id)
        return features

    def forward(self, x, expert_id, inference = False):
        logits = []
        features = self.backbone(x, expert_id = expert_id)

        if inference:
            for prompts in self.classifier_pool[:self._cur_task_id + 1]:
                logits.append(prompts(features))
        else:
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits.append(prompts(features))

        return torch.cat(logits, dim=1)

    def update_input_matrix(self, x, expert_id):
        self.backbone(x, expert_id = expert_id, get_input_matrix = True)

class MoE_Test(nn.Module):

    def __init__(self, backbone, device, **kwargs):
        super().__init__()

        self.device = device
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.lame = kwargs["lame"]
        self.lamb = kwargs["lamb"]

        self._known_classes = 0
        self.feature_list = []
        self.project_type = []

        self._network = Model(backbone, **kwargs)

        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_MoEMaskedLoRA)]

        # TRGP Implementation
        self.feature_list_each_tasks = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]
        self.final_decision = [[np.zeros((1)) for _ in range(len(self.attention_modules))] for _ in range(self.task_num)]

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
    
        self._network.to(self.device)
        
    def observe(self, data):
        '''
        Called during the training phase, it inputs a batch of training examples and returns the prediction, accuracy, and forward loss.
        '''

        x, y = data['image'].to(self.device), data['label'].to(self.device) - self._known_classes

        logits = self._network(x, expert_id = self._network._cur_task_id) # hardcoded for task_id
        loss = F.cross_entropy(logits, y)

        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc, loss
    
    def inference(self, data):

        x, y = data['image'].to(self.device), data['label'].to(self.device)

        logits = self._network(x, expert_id = 0, inference = True)
        preds = logits.max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        return preds, acc
    
    @torch.no_grad()
    def before_task(self, task_idx, buffer, train_loader, _):

        if task_idx == 1:
            self._known_classes += self.init_cls_num
        elif task_idx > 1:
            self._known_classes += self.inc_cls_num
        self._network.update_fc()

        for module in self.attention_modules:
            module.init_param()

        for batch in tqdm(train_loader, desc = "Forwarding to get input matrix"):
            x = batch['image'].to(self.device)
            self._network.update_input_matrix(x, expert_id = 0)

        if task_idx == 0:
            for i, module in enumerate(self.attention_modules):
                U, _, _ = torch.linalg.svd(module.cur_matrix)
                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))

                #module.lora_A_k_ts[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                #module.lora_A_v_ts[task_idx].weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.reset_input_matrix()
        else:
            for i, module in enumerate(self.attention_modules):
                assert self.project_type[i] == 'remove' or self.project_type[i] == 'retain'

                cur_matrix = module.cur_matrix
                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T)

                U, _, _ = np.linalg.svd(cur_matrix.cpu().numpy(), full_matrices = False)
                U = torch.tensor(U).to(self.device)

                #module.lora_A_k_ts[task_idx].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                #module.lora_A_v_ts[task_idx].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))

                if self.project_type[i] == 'remove':
                    cur_matrix = cur_matrix - feature_mat @ cur_matrix
                else:
                    cur_matrix = feature_mat @ cur_matrix

                U, _, _ = np.linalg.svd(cur_matrix.cpu().numpy(), full_matrices = False)
                U = torch.tensor(U).to(self.device)

                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.reset_input_matrix()

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)            

            if f'classifier_pool.{task_idx}' in name or \
               (f'lora_B' in name and 'ts' not in name) or \
               f"B_k_ts.{task_idx}" in name or \
               f"B_v_ts.{task_idx}" in name or \
               f"scale_param.{task_idx}" in name or \
               f'eye' in name or \
               f'router' in name or \
               f'w_noise' in name:
            
                param.requires_grad_(True)
        
        unfrezeed_params = [name for name, param in self._network.named_parameters() if param.requires_grad]
        print('\n'.join(unfrezeed_params))

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        for module in self.attention_modules:
            module.merge_weight()

        self._update_feature(task_idx, train_loader)

        #self._network.eval()
        #self._create_distribution(task_idx, train_loader, test_loaders[0].dataset.trfms) # also compute class mean here
        #
        #if task_idx > 0 and self._use_class_alignment:
        #    self._compact_classifier(task_idx)

    @torch.no_grad()
    def _update_feature(self, task_idx, train_loader):
        '''
        Update feature lists and the corresponding type
        '''

        for batch in tqdm(train_loader, desc="Forwarding to get input matrix"):
            x = batch['image'].to(self.device)
            self._network.update_input_matrix(x, expert_id = 0)

        threshold = (self.lame - self.lamb)*task_idx/self.task_num + self.lamb

        if task_idx == 0:
            for i, attention_module in enumerate(self.attention_modules):
                activation = attention_module.cur_matrix

                U, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = max(np.sum(np.cumsum(sval_ratio) < threshold), 1)
                assert r < activation.shape[0]/2

                self.feature_list_each_tasks[task_idx][i] = U[:, :r]
                self.feature_list.append(U[:, :r])
                self.project_type.append('remove')

                attention_module.reset_input_matrix()                
        else:
            for i, attention_module in enumerate(self.attention_modules):

                activation = attention_module.cur_matrix
                _, S, _ = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()

                if self.project_type[i] == 'remove':

                    act_hat = activation - torch.Tensor(self.feature_list[i] @ self.feature_list[i].transpose()) @ activation
                    U, S, _ = np.linalg.svd(act_hat, full_matrices = False)
                    sigma = S**2

                    delta = (torch.tensor(self.feature_list[i]).T @ activation @ activation.T @ torch.tensor(self.feature_list[i])).diagonal()

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

                        
                else:
                    act_hat = Torch.Tensor(self.feature_list[i] @ self.feature_list[i].transpose()) @ activation
                    U,S,_ = np.linalg.svd(act_hat, full_matrices = False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total     
                    accumulated_sval = sval_hat/sval_total          

                    if accumulated_sval < 1 - threshold:
                        print (f'Skip Updating Space for layer: {i+1}')
                    else:
                        r = np.sum(accumulated_sval - np.cumsum(sval_ratio) >= 1 - threshold) + 1
                        act_feature = self.feature_list[i] - U[:,0:r] @ U[:,0:r].T @ self.feature_list[i]
                        U, _, _ = np.linalg.svd(act_feature)
                        self.feature_list[i]=U[:,:self.feature_list[i].shape[1]-r]

                attention_module.reset_input_matrix()

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
    def _create_distribution(self, task_idx, train_loader, test_trfms):

        train_loader.dataset.trfms = test_trfms

        features = []
        for batch in train_loader:
            x = batch['image'].to(self.device)
            features.append(self._network.get_feature(x, expert_id = 0))
            features.append(self._network.get_feature(torch.flip(x, dims=(3,)), expert_id = 0))
        features = torch.cat(features, dim = 0)

        eps = 1e-8
        while True:
            try:
                assert eps < 1, 'eps too high, either wrong backbone implementation or gpu out of memory'
                gmm = GaussianMixture(1, features.shape[1], covariance_type='full', eps=eps).to(self.device)
                gmm.fit(features, delta=1e-3, n_iter=100)
                gmm.mu.data = gmm.mu.data.unsqueeze(1)
                break
            except RuntimeError:
                eps *= 10
                print(f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")

        self.experts_distributions.append(gmm)

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Assuming samples_1 and samples_2 are already defined and contain your sample data
        samples = []
        for gmm in self.experts_distributions:
            sample, _ = gmm.sample(1000)
            samples.append(sample.cpu().numpy())

        colors = sns.color_palette("husl", len(samples))  # Use Seaborn's color palette

        # Create the plot
        plt.figure(figsize=(8, 8))

        # Iterate through each sample and plot it with a unique color
        for i, sample in enumerate(samples):
            sns.scatterplot(x=sample[:, 0], y=sample[:, 1], alpha=0.5, color=colors[i], label=f'Samples {i + 1}')

        # Add title and labels
        plt.title('2D Projection of Multiple Sample Distributions')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        # Save the figure as a PNG file
        #plt.savefig('2d_projection_distribution_multiple.png')
        #plt.close()  # Close the figure after saving

        if self._use_class_alignment:
            samples = [[] for _ in range(self.inc_cls_num)]
            for batch in train_loader:
                x, y = batch['image'], batch['label'] - self._known_classes
                for label in range(self.inc_cls_num):
                    samples[label].append(x[y == label])
            samples = [torch.cat(label_sample, dim = 0).to(self.device) for label_sample in samples]

            # Computing class mean
            if self._class_means is None:
                self._class_means = torch.zeros((self.init_cls_num, 768))
                self._class_covs = torch.zeros((self.init_cls_num, 768, 768))
            else:
                self._class_means = torch.cat((self._class_means, torch.zeros((self.inc_cls_num, 768))), dim=0)
                self._class_covs = torch.cat((self._class_covs, torch.zeros((self.inc_cls_num, 768, 768))), dim=0)

            for class_idx, x in enumerate(samples):
                class_idx += self._known_classes
                features = self._network.get_feature(x, expert_id = task_idx)

                self._class_means[class_idx, :] = torch.mean(features, dim = 0)
                self._class_covs[class_idx, :, :] = torch.cov(features.to(torch.float64).T) + torch.eye(768, device = self.device) * 1e-4

    def _compact_classifier(self, task_idx):

        # Hyperparam
        epoch = 5
        lr = 0.01
        weight_decay = 0.0005
        momentum = 0.9
        num_sample = 256

        for param in self._network.classifier_pool[:task_idx + 1].parameters():
            param.requires_grad_(True)
        param_list = [param for param in self._network.classifier_pool.parameters() if param.requires_grad]

        optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch)

        for ep in range(epoch):
            sampled_data, sampled_label = [], []

            for class_id in range((task_idx + 1) * self.inc_cls_num):
                task_id = class_id // self.inc_cls_num

                decay = (task_id + 1) / (task_idx + 1) * 0.1
                cls_mean = self._class_means[class_id].to(self.device, torch.float64) * (0.9 + decay)
                cls_cov = self._class_covs[class_id].to(self.device)

                m = torch.distributions.multivariate_normal.MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sample,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([class_id] * num_sample)

            inputs = torch.cat(sampled_data, dim=0).float().to(self.device)
            targets = torch.tensor(sampled_label).long().to(self.device)

            # Randomize
            #sf_indexes = torch.randperm(inputs.size(0))
            #inputs = inputs[sf_indexes]
            #targets = targets[sf_indexes]
            
            for _iter in range((task_idx + 1) * self.inc_cls_num):
                
                task_id = _iter // self.inc_cls_num

                inp = inputs[_iter*num_sample:(_iter+1)*num_sample]
                tgt = targets[_iter*num_sample:(_iter+1)*num_sample]
                logits = self._network.fc_only(inp, task_id)

                #print('logits.shape', logits.shape)
                #assert 0
                assert logits.shape == logits[:, :(task_idx + 1) * self.inc_cls_num].shape

                loss = F.cross_entropy(logits, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

    def _set_random(self,args):
        '''
        Set random values on various devices to ensure repeatable results
        '''
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_parameters(self, config):
        return self._network.parameters()        