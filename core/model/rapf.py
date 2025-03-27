import copy

import torch
import torch.nn as nn
import numpy as np
import os
import random

from tqdm import tqdm

from .backbone.clip import tokenize
from core.data import dataloader
from core.model import backbone
from core.model.finetune import Finetune
from torch.utils.data import DataLoader


def get_class_ids_per_task(init_cls_num, inc_cls_num, class_order):
    yield class_order[:init_cls_num]
    for i in range(init_cls_num, len(class_order), inc_cls_num):
        yield class_order[i:i + inc_cls_num]

def get_class_names(classes_names, prev_cls_num, accu_cls_num):
    return [classes_names[i] for i in range(prev_cls_num, accu_cls_num)]

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


"""
This clas refer to the following repository:
https://github.com/linlany/RAPF
"""
class ClassIncrementalCLIP(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        device = kwargs['device']
        fp16 = kwargs['fp16'] 
        mix_bias = kwargs['mix_bias'] 
        self.prompt_template = kwargs['prompt_template']
        self.initial_increment = kwargs['init_cls_num'] 
        self.increment = kwargs['inc_cls_num']
        self.device = device
        self.classes_names = None
        # self.class_order = kwargs['class_order']
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # pdb.set_trace()
        # self.class_ids_per_task = list(get_class_ids_per_task(self.initial_increment, self.increment, self.class_order))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if fp16 else torch.float32
        self.adapter = nn.Linear(512, 512, bias=False ,device=device)
        self.clip_type = model.dtype


        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = mix_bias

    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text).type(self.clip_type)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_image(self, image):
         # 确保输入数据类型与 self.visual 的权重类型一致
        image = image.to(self.clip_type)
        return self.visual(image)

    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        image = image.type(torch.float16)
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)


        with torch.no_grad():
            image_features = self.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.type(self.dtype).detach()).type(self.clip_type)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)
        
        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def adaptation(self, task_id, prev_cls_num, accu_cls_num, threshold=0):
        self.current_class_names += get_class_names(self.classes_names, prev_cls_num, accu_cls_num)
        self.text_tokens = tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id>0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:prev_cls_num]):
                diff = torch.cdist(self.class_name_features[prev_cls_num:].type(torch.float32), class_name_feature.unsqueeze(0).type(torch.float32)).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_new_class = torch.unique(indices[:,1]) + self.initial_increment+(task_id-1) * self.increment
            num_hard_class = self.hard_new_class.shape[0]
            self.hard_pairs = indices
            self.hard_pairs[:,1] = self.hard_pairs[:,1]+self.initial_increment+(task_id-1) * self.increment
    def get_old_edge_samples(self, batch_size):
        random_select = torch.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], self.old_edge_samples_nearest_labels[random_select]


    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4* torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean()-max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            dist = (weight_new - weight_old).abs()
            U_old, S_old, V_old = torch.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = (P_new - torch.diag(S_old)@V_old).abs()
            mask = dist / dist.max()
            mask += self.mix_b
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + torch.diag(S_old)@V_old * (1-mask)
            weight = U_old @ right
            self.adapter.weight.data = weight

"""
This clas refer to the following repository:
https://github.com/linlany/RAPF
"""
class RAPF(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        seed = kwargs['seed']
        seed_everything(seed) 
        self.backbone = backbone
        self.kwargs = kwargs
        self.model = ClassIncrementalCLIP(self.backbone, **kwargs)
        self.device = kwargs['device']
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num = kwargs['inc_cls_num']
        self.beta = kwargs['beta']
        self.shrinkage = kwargs['shrinkage']
        self.threshold = kwargs['threshold']
        self.train_batch_size = kwargs['train_batch_size']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.seed = seed

        self.prev_cls_num = 0
        self.accu_cls_num = 0



    def before_task(self, task_id, buffer, train_loader, test_loaders):
        self.task_id = task_id
        if self.task_id == 0:
            self.accu_cls_num = self.init_cls_num
        else:
            self.accu_cls_num += self.inc_cls_num

        self.model.adaptation(task_id, self.prev_cls_num, self.accu_cls_num, self.threshold)
        if self.task_id > 0:
            random_class_order_list = list(range(self.init_cls_num+(self.task_id-1)*self.inc_cls_num))
            random.shuffle(random_class_order_list)
            self.random_class_order_list = random_class_order_list

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        model = self.model
        for batch in tqdm(train_loader, total=len(train_loader)):
            feats = batch['image']
            target = batch['label']
            feats, target = feats.to(self.device), target.to(self.device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature = model(feats, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        sample_after_adapt_feature = torch.cat(sample_after_adapt_feature, dim=0)
        model.analyze_mean_cov(sample_data, sample_target)
        model.mix_matrix()
        self.prev_cls_num = self.accu_cls_num

    def get_parameters(self, config):
        return self.model.adapter.parameters()

    def observe(self, data):
        loss = torch.tensor(0.0).to(self.device)
        loss_c = torch.tensor(0.0).to(self.device)
        loss_hinge = torch.tensor(0.0).to(self.device)
        
        inputs = data['image']
        targets = data['label']
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        sg_inputs = None
        edge_sample = None
        ori_targets = targets.clone()
        model = self.model
        if self.task_id > 0:
            sg_inputs = []
            sg_targets = []
            # num of classes per batch. Ensure an epoch traverses all classes at least once. 
            # For exemple, if there are 100 classes and 50 batches per epoch , there will be 2 classes per batch.

            random_class_order_list = self.random_class_order_list
            batch_id = data['batch_id'] 
            if self.inc_cls_num == 5:
                list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
            else:
                list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]

            
            for i in list_for_one_batch:
                sg_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10*self.beta), shrink=self.shrinkage))
                sg_targets.append(torch.ones(int(10*self.beta), dtype=torch.long, device=self.device)*i)
            sg_inputs = torch.cat(sg_inputs, dim=0)
            sg_targets = torch.cat(sg_targets, dim=0)
            targets = torch.cat([targets, sg_targets], dim=0)
        if model.hard_pairs is not None and model.hard_pairs.shape[0] > 0:
            edge_sample = []
            edge_p_target = []
            edge_n_target = []
            for hard_pair in model.hard_pairs:
                edge_sample.append(sample(model.class_mean_list[hard_pair[0]], model.class_cov_list[hard_pair[0]],int(20*self.beta), shrink=self.shrinkage))
                edge_p_target.append(torch.ones(int(20*self.beta), dtype=torch.long, device=self.device)*hard_pair[0])
                edge_n_target.append(torch.ones(int(20*self.beta), dtype=torch.long, device=self.device)*hard_pair[1])
            edge_sample = torch.cat(edge_sample, dim=0)
            edge_p_target = torch.cat(edge_p_target, dim=0)
            edge_n_target = torch.cat(edge_n_target, dim=0)
        if self.task_id > 0:
            not_ini = True
        else:
            not_ini = False
        outputs, _, __, edge_sample_features = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample, prompt=False)

        if self.task_id > 0:
            if edge_sample is not None:
                edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                loss_hinge = torch.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
        loss_c = torch.nn.functional.cross_entropy(outputs, targets.detach())
        if edge_sample is not None:
            loss = loss_c + loss_hinge
        else:
            loss = loss_c 
        # Return tuple [pred, acc, loss]
        # with torch.no_grad():
            # prob_outputs = torch.nn.functional.softmax(outputs, dim=-1)
        predicted_labels = outputs.argmax(dim=1)
        predicted_labels = predicted_labels[:ori_targets.size(0)]
        corrects = (predicted_labels == ori_targets).sum().item()
        total_predictions = ori_targets.size(0)
        accuracy = corrects / total_predictions
        return predicted_labels, accuracy, loss


    def inference(self, data):
        feats = data['image']
        target = data['label']
        feats, target = feats.to(self.device), target.to(self.device)
        model = self.model
        with torch.no_grad():
            outputs, _, __, ___ = model(feats, prompt=False)
            prob_outputs = torch.nn.functional.softmax(outputs, dim=-1)
            predicted_labels = prob_outputs.argmax(dim=1)
            corrects = (predicted_labels == target).sum().item()
            total_predictions = target.size(0)
            accurcy = corrects / total_predictions
        return prob_outputs, accurcy