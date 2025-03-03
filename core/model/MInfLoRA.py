"""
Code Reference:
https://github.com/liangyanshuo/InfLoRA/blob/main/methods/inflora.py
"""
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

from .backbone.transformer import MultiHeadAttention_MaskedLoRA

Epsilon = 0.5

class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            self.mu.data = self.get_kmeans_mu(x, n_centers=self.n_components)

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze(0).squeeze(1)).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in torch.arange(0, self.n_components, device=counts.device)[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return self.__score(x, as_average=False)

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":

            log_2pi = x.shape[-1] * np.log(2. * pi)

            precision = torch.inverse(self.var)
            log_det = self._calculate_log_det(precision)

            #log_det = self._calculate_log_det(self.var)

            #if -log_det != log_det_pos:
            #    print(f'not same : {log_det}, {log_det_pos}')
            #else:
            #    print(f'same : {log_det}, {log_det_pos}')

            x_mu_T = (x - self.mu).unsqueeze(-2)
            x_mu = (x - self.mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            #return -.5 * (log_2pi + log_det + x_mu_T_precision_x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """

        log_det = torch.empty(size=(self.n_components,)).to(var.device) # (k, )

        #for k in range(self.n_components):
        #    log_det[k] = torch.linalg.cholesky(var[0, k]).diagonal().log().sum()

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        #return log_det.unsqueeze(-1) * 2 
        return log_det.unsqueeze(-1) 

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires dea threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res

def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)

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

        self.classifier_pool = nn.ModuleList([
            nn.Linear(kwargs["embd_dim"], kwargs['init_cls_num'], bias=True)] + 
            [nn.Linear(kwargs["embd_dim"], kwargs['inc_cls_num'], bias=True) for _ in range(kwargs['task_num'] - 1)])

        #for layer in self.classifier_pool:
        #    nn.init.xavier_uniform_(layer.weight)
        #    nn.init.constant_(layer.bias, 0)

    def update_fc(self):
        self._cur_task_id += 1

    def fc_only(self, x, expert_id):
        logits = []
        for prompts in self.classifier_pool[:expert_id + 1]:
            logits.append(prompts(x))
        return torch.cat(logits, dim=1)
    
    def fc_only2(self, x):
        logits = []
        for prompts in self.classifier_pool[:self._cur_task_id + 1]:
            logits.append(prompts(x))
        return torch.cat(logits, dim=1)

    def get_feature(self, x, expert_id):
        features = self.backbone(x, expert_id = expert_id)
        return features

    def forward(self, x, expert_id, inference = False):
        logits = []
        features = self.backbone(x, expert_id = expert_id)

        if inference:

            for prompts in self.classifier_pool[:expert_id+1]:
                logits.append(prompts(features))
            if expert_id < self._cur_task_id:
                for prompts in self.classifier_pool[expert_id+1:self._cur_task_id+1]:
                    logits.append(torch.full_like(prompts(features), 1e-10))

        else:
            for prompts in [self.classifier_pool[self._cur_task_id]]:
                logits.append(prompts(features))

        return torch.cat(logits, dim=1)

    # bayesian only
    def inference(self, x, expert_id):
        logits = []
        features = self.backbone(x, expert_id = expert_id)
        for prompts in self.classifier_pool[:self._cur_task_id + 1]: 
                logits.append(prompts(features))
        return torch.cat(logits, dim=1)

    def update_input_matrix(self, x, expert_id):
        self.backbone(x, expert_id = expert_id, get_input_matrix = True)

class MInfLoRA(nn.Module):

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

        self._network = SiNet(backbone, **kwargs)

        self.attention_modules = [module for module in self._network.modules() if isinstance(module, MultiHeadAttention_MaskedLoRA)]

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
    
    def inference(self, data, **kwargs):

        task_id = kwargs['task_id'] if 'task_id' in kwargs else None 
        x, y = data['image'].to(self.device), data['label'].to(self.device)

        features = self._network.get_feature(x, expert_id = 0)
        log_probs = torch.zeros((features.shape[0], len(self.experts_distributions)), device = self.device)

        for task, task_gmm in enumerate(self.experts_distributions):
            log_probs[:, task] = task_gmm.score_samples(features)

        temperature = y.shape[0] * 15
        prob = F.softmax(torch.sum(log_probs / temperature, dim = 0), dim = 0)
        prob = prob / ((1.5) ** torch.arange(len(self.experts_distributions))).to(self.device)
        pred_task = prob.argmax()

        logits = self._network.inference(x, expert_id=pred_task)
        preds = (logits.softmax(dim=1) * prob.repeat_interleave(self.inc_cls_num)).max(1)[1]
        acc = preds.eq(y).sum().item() / y.shape[0]

        if len(self.experts_distributions) > 1:
            top_values, _ = torch.topk(prob, k=2)
            sorted_top_values, _ = torch.sort(top_values, descending=True)
            min_value = torch.min(prob)
            confidence = sorted_top_values[0] - sorted_top_values[1]
            confidence3 = confidence / (sorted_top_values[0] - min_value.item())
        else:
            return preds, acc

        if task_id: 

            if pred_task != task_id:
                print('-'*15)
                print(f'Miss : {task_id}/{pred_task} with confidence {confidence:.2f}, {confidence3:.2f}')

            else:
                print('-'*15)
                print(f'Hit : {task_id}/{pred_task} with confidence {confidence:.2f}, {confidence3:.2f}')
                print('-'*15)

        else:
            print('-'*15)
            print(f'{pred_task} with confidence {confidence:.2f}, {confidence3:.2f}')
            print('-'*15)

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

        for i, module in enumerate(self.attention_modules):
            
            topk = TopK(1)

            for task_id in range(task_idx):
            
                mat = module.cur_matrix.cpu().numpy()
                proj_norm = np.linalg.norm(self.feature_list_each_tasks[task_id][i] @ self.feature_list_each_tasks[task_id][i].T @ mat)
                mat_norm = np.linalg.norm(mat)

                if proj_norm > Epsilon * mat_norm:
                    topk.add({'proj_norm':proj_norm, 'task_id': task_id})

                #print(f'Layer {i} of {task_idx} to {task_id} : {proj_norm:.4f}/{mat_norm:.4f} ({proj_norm > Epsilon * mat_norm})')

            self.final_decision[task_idx][i] = [dic['task_id'] for dic in topk.get_top_k()]

            module.enable_scale(task_id = task_idx, space = [torch.tensor(self.feature_list_each_tasks[task_id][i]).to(self.device) for task_id in self.final_decision[task_idx][i]])
            print(f'Layer {i} of {task_idx} consider {self.final_decision[task_idx][i]} as trust region')

        if task_idx == 0:
            for i, module in enumerate(self.attention_modules):
                U, _, _ = torch.linalg.svd(module.cur_matrix)
                module.lora_A_k.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.lora_A_v.weight.data.copy_(U[:,:module.lora_rank].T/math.sqrt(3))
                module.reset_input_matrix()
        else:
            for i, module in enumerate(self.attention_modules):
                assert self.project_type[i] == 'remove' or self.project_type[i] == 'retain'

                cur_matrix = module.cur_matrix
                feature_mat = torch.Tensor(self.feature_list[i] @ self.feature_list[i].T)

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
            if f"classifier_pool.{task_idx}" in name or f"lora_B" in name or f"scale_param.{task_idx}" in name or 'eye' in name:
                param.requires_grad_(True)
        unfrezeed_params = [name for name, param in self._network.named_parameters() if param.requires_grad]

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
        Called after each task before final testing, it is used to perform preliminary operations on the mapping matrix to facilitate the update of lora_a layer in the next round of before_task
        '''

        for module in self.attention_modules:
            module.merge_weight()

        self._update_feature(task_idx, train_loader)

        self._network.eval()
        self._create_distribution(task_idx, train_loader, test_loaders[0].dataset.trfms) # also compute class mean here
        
        if task_idx > 0 and self._use_class_alignment:
            self._compact_classifier(task_idx)

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