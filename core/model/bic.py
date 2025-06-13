"""
@inproceedings{wu2019large,
  title={Large Scale Incremental Learning},
  author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={374--382},
  year={2019}
}
https://arxiv.org/abs/1905.13260

Adapted from https://github.com/wuyuebupt/LargeScaleIncrementalLearning and https://github.com/sairin1202/BIC.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch import nn
from copy import deepcopy
from torch.utils.data import DataLoader
from core.model.backbone.resnet import BiasLayer
from collections import Counter

# Spilt images and labels into train_dataset and test_dataset, it is assured that each label's count are balanced in both output dataset
def balance_spilt(images, labels, test_size, random_state=None):

    images, labels = np.array(images), np.array(labels)

    classes = np.unique(labels)
    expected_class_count = len(labels) // len(classes)

    for label, count in Counter(labels).items():
        assert count == expected_class_count, f'Label {label} has only {count}, expected to be {expected_class_count}'
    
    train_count = int(expected_class_count * (1 - test_size))
    test_count = int(expected_class_count * test_size)
    assert train_count + test_count == int(expected_class_count)

    selected_train_images, selected_train_labels = [], []
    selected_val_images, selected_val_labels = [], []

    for class_id in classes:
        idx, = np.where(np.array(labels) == class_id)

        if random_state:
            np.random.seed(random_state)
            np.random.shuffle(idx)

        cls_images, cls_labels = images[idx], labels[idx]

        selected_train_images.extend(cls_images[:train_count])
        selected_train_labels.extend(cls_labels[:train_count])

        selected_val_images.extend(cls_images[train_count:])
        selected_val_labels.extend(cls_labels[train_count:])

        assert len(cls_images[:train_count]) == train_count
        assert len(cls_labels[:train_count]) == train_count
        assert len(cls_images[train_count:]) == test_count
        assert len(cls_labels[train_count:]) == test_count
        
    return selected_train_images, selected_val_images, selected_train_labels, selected_val_labels

# Simply spilt the dataset by slicing, the balance of label is not assured
def slice_spilt(images, labels, test_size):

    total = len(labels)

    train_count = int(total*(1-test_size))
    test_count = int(total*test_size)

    selected_train_images, selected_train_labels = images[:train_count], labels[:train_count]
    selected_val_images, selected_val_labels = images[train_count:], labels[train_count:]

    return selected_train_images, selected_val_images, selected_train_labels, selected_val_labels

class Model(nn.Module):

    def __init__(self, backbone, num_class, device):
        super().__init__()
        self.backbone = backbone
        self.num_class = num_class
        self.classifier = nn.Linear(backbone.feat_dim, num_class)
    
    def forward(self, x):
        return self.classifier(self.backbone(x))
    
class bic(nn.Module):
    def __init__(self, backbone, num_class, **kwargs):

        super().__init__()

        self.device = kwargs['device']
        self.task_num = kwargs['task_num']
        self.bias_layers = nn.ModuleList([BiasLayer().to(self.device) for _ in range(self.task_num)])

        params = []
        for layer in self.bias_layers:
            params += layer.parameters()

        self.bias_optimizer = optim.Adam(params, lr = 1e-3)
        self.model = Model(backbone, num_class, self.device)
        self.init_cls_num = kwargs['init_cls_num']
        self.inc_cls_num  = kwargs['inc_cls_num']

        self.seen_cls = 0
        self.cur_task = 0

        self.previous_model = None
        self.criterion = nn.CrossEntropyLoss()
 
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        
        self.previous_model = deepcopy(self.model)

        for param in self.previous_model.parameters():
            param.requires_grad_(False)

        for param in self.model.parameters():
            param.requires_grad_(True)

        for layer in self.bias_layers:
            [param.requires_grad_(False) for param in layer.parameters()]

        self.cur_task = task_idx
        self.seen_cls += self.init_cls_num if task_idx == 0 else self.inc_cls_num

    def bias_forward(self, input, train=True):

        outputs = []

        train = False
        if train:

            for i, layer in enumerate(self.bias_layers):
                if i == 0:
                    input_slice = input[:, :self.init_cls_num]
                else:
                    input_slice = input[:, (i-1) * self.inc_cls_num + self.init_cls_num : i * self.inc_cls_num + self.init_cls_num]
                
                if i == self.cur_task:
                    outputs.append(layer(input_slice))
                else:
                    outputs.append(input_slice)

        else:

            for i, layer in enumerate(self.bias_layers):
                if i == 0:
                    input_slice = input[:, :self.init_cls_num]
                else:
                    input_slice = input[:, (i-1) * self.inc_cls_num + self.init_cls_num : i * self.inc_cls_num + self.init_cls_num]

                outputs.append(layer(input_slice))

        return torch.cat(outputs, dim=1)

    def inference(self, data):
        x, y = data['image'].to(self.device), data['label'].view(-1).to(self.device)
        
        p = self.model(x)
        p = self.bias_forward(p, train = False)
        pred = p[:, :self.seen_cls].argmax(dim=-1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0)
     
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        
        for param in self.model.parameters():
            param.requires_grad_(False)

        for i, layer in enumerate(self.bias_layers):

            if i == task_idx:
                [param.requires_grad_(True) for param in layer.parameters()]
            else:
                [param.requires_grad_(False) for param in layer.parameters()]

            print(f'Bias Layer {i} : {layer.alpha.item()}, {layer.beta.item()} {layer.alpha.requires_grad}')

    # The classic two-phase processing approach employed by BIC.
    def stage1(self, data):

        x, y = data['image'].to(self.device), data['label'].view(-1).to(self.device)

        p = self.model(x)
        
        p = self.bias_forward(p)
        loss = self.criterion(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        return pred, acc / x.size(0), loss

    def stage1_distill(self, data):

        x, y = data['image'].to(self.device), data['label'].view(-1).to(self.device)

        T = 2 # temperature
        alpha = 1.0 * (self.seen_cls - self.inc_cls_num) / self.seen_cls
        assert 1.0 * self.cur_task / (self.cur_task + 1) == alpha
        
        p = self.model(x)
        p = self.bias_forward(p)

        pred = torch.argmax(p[:, :self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        with torch.no_grad():
            pre_p = self.previous_model(x)
            pre_p = self.bias_forward(pre_p, train = True)
            pre_p = F.softmax(pre_p[:, :self.seen_cls - self.inc_cls_num]/T, dim=1)

        logp = F.log_softmax(p[:, :self.seen_cls-self.inc_cls_num]/T, dim=1)
        loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
        loss_hard_target = self.criterion(p[:, :self.seen_cls], y)
        loss = alpha * loss_soft_target * T * T + (1-alpha) * loss_hard_target # T**2 stated in 'Distilling the Knowledge in a Neural Network', last paragraph of section 'Distillation'
    
        return pred, acc / x.size(0), loss

    def stage2(self, data):

        x, y = data['image'].to(self.device), data['label'].view(-1).to(self.device)
        p = self.model(x)
        p = self.bias_forward(p)
        loss = self.criterion(p[:,:self.seen_cls], y)
        pred = torch.argmax(p[:,:self.seen_cls], dim=1)
        acc = torch.sum(pred == y).item()

        self.bias_optimizer.zero_grad()
        loss.backward()
        self.bias_optimizer.step()

        return pred, acc / x.size(0), loss

    def observe(self, data):

        if self.cur_task > 0:
            return self.stage1_distill(data)
        else:
            return self.stage1(data)

    def get_parameters(self, config):

        return self.model.parameters()

    # split_and_update1 （比例，且多）: 将新的训练数据分成 9:1, 按照未更新的 buffer 中的 val data 的数量加入 val data
    # split_and_update2 （比例，但少）: 将新的训练数据分成 9:1, 按照更新后的 buffer 中的 val data 的数量加入 val data
    # split_and_update4 （非比例，且少）: 根据未更新的 buffer 中的 val data 的数量分割新的训练数据，然后安排
    
    @staticmethod
    def spilt_and_update1(dataloader, buffer, task_idx, config):

        print('using spilt_and_update1')
        
        train_dataset = deepcopy(dataloader.dataset)
        val_dataset = deepcopy(dataloader.dataset)

        # Train_loader
        images_train, images_val, labels_train, labels_val = slice_spilt(
            train_dataset.images,
            train_dataset.labels,
            test_size=0.1
        )

        train_dataset.images = images_train + buffer.train_images
        train_dataset.labels = labels_train + buffer.train_labels

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,  
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True)

        # Val_loader
        if task_idx == 0:
            val_dataloader = None
        else:
            current_num_per_classes_val = min(len(labels_val)//config['inc_cls_num'], (buffer.buffer_size * 0.1) // buffer.total_classes)
            print(f'Assigning {current_num_per_classes_val} per class in val data')

            current_num_per_classes_val = int(current_num_per_classes_val)

            selected_images_val, selected_labels_val = [], []

            for cls_label in np.unique(labels_val):
                
                cls_idx, = np.where(np.array(labels_val) == cls_label)
                cls_images, cls_labels = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

                selected_images_val.extend(cls_images[:current_num_per_classes_val])
                selected_labels_val.extend(cls_labels[:current_num_per_classes_val])

                print(f'{cls_label}, {len(cls_labels[:current_num_per_classes_val])}/{len(cls_labels)}')

            print(buffer.val_labels)

            for cls_label in range(buffer.total_classes):
                cls_idx, = np.where(np.array(buffer.val_labels) == cls_label)
                cls_images, cls_labels = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

                selected_images_val.extend(cls_images[:current_num_per_classes_val])
                selected_labels_val.extend(cls_labels[:current_num_per_classes_val])

                print(f'{cls_label}, {len(cls_labels[:current_num_per_classes_val])}/{len(cls_labels)}')

            val_dataset.images = selected_images_val
            val_dataset.labels = selected_labels_val

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=True,  
                batch_size=100,
                num_workers=config['num_workers'],
                drop_last=False)

        # Update Buffer
        buffer.total_classes += config['init_cls_num'] if task_idx == 0 else config['inc_cls_num']
        new_num_per_classes_train = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.9)
        new_num_per_classes_val = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.1)

        preserved_images_train, preserved_labels_train = [], []
        preserved_images_val, preserved_labels_val = [], []

        # Preserved old in buffer
        for old_cls_label in np.unique(buffer.train_labels):
            
            cls_idx = np.where(buffer.train_labels == old_cls_label)
            cls_images_train, cls_labels_train = np.array(buffer.train_images)[cls_idx], np.array(buffer.train_labels)[cls_idx]

            cls_idx = np.where(buffer.val_labels == old_cls_label)
            cls_images_val, cls_labels_val = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

        # Add new into buffer
        for new_cls_label in np.unique(labels_train):
            
            cls_idx = np.where(labels_train == new_cls_label)
            cls_images_train, cls_labels_train = np.array(images_train)[cls_idx], np.array(labels_train)[cls_idx]

            cls_idx = np.where(labels_val == new_cls_label)
            cls_images_val, cls_labels_val = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

        buffer.train_images = preserved_images_train
        buffer.train_labels = preserved_labels_train
        buffer.val_images = preserved_images_val
        buffer.val_labels = preserved_labels_val

        return train_dataloader, val_dataloader

    @staticmethod
    def spilt_and_update2(dataloader, buffer, task_idx, config):

        print('using spilt_and_update2')
        
        train_dataset = deepcopy(dataloader.dataset)
        val_dataset = deepcopy(dataloader.dataset)

        # Train_loader
        images_train, images_val, labels_train, labels_val = slice_spilt(
            train_dataset.images,
            train_dataset.labels,
            test_size=0.1
        )

        train_dataset.images = images_train + buffer.train_images
        train_dataset.labels = labels_train + buffer.train_labels

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,  
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True)

        # Update Buffer
        buffer.total_classes += config['init_cls_num'] if task_idx == 0 else config['inc_cls_num']
        new_num_per_classes_train = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.9)
        new_num_per_classes_val = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.1)

        preserved_images_train, preserved_labels_train = [], []
        preserved_images_val, preserved_labels_val = [], []

        # Preserved old in buffer
        for old_cls_label in np.unique(buffer.train_labels):
            
            cls_idx = np.where(buffer.train_labels == old_cls_label)
            cls_images_train, cls_labels_train = np.array(buffer.train_images)[cls_idx], np.array(buffer.train_labels)[cls_idx]

            cls_idx = np.where(buffer.val_labels == old_cls_label)
            cls_images_val, cls_labels_val = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

            print(f'{old_cls_label}, {len(cls_labels_val[:new_num_per_classes_val])}/{len(cls_labels_val)}')
        
        # Add new into buffer
        for new_cls_label in np.unique(labels_train):
            
            cls_idx = np.where(labels_train == new_cls_label)
            cls_images_train, cls_labels_train = np.array(images_train)[cls_idx], np.array(labels_train)[cls_idx]

            cls_idx = np.where(labels_val == new_cls_label)
            cls_images_val, cls_labels_val = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

            print(f'{new_cls_label}, {len(cls_labels_val[:new_num_per_classes_val])}/{len(cls_labels_val)}')

        buffer.train_images = preserved_images_train
        buffer.train_labels = preserved_labels_train
        buffer.val_images = preserved_images_val
        buffer.val_labels = preserved_labels_val

        # Val loader
        if task_idx == 0:
            val_dataloader = None
        else:
            print(f'Assigning {new_num_per_classes_val} per class in val data')

            val_dataset.images = buffer.val_images
            val_dataset.labels = buffer.val_labels

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=True,  
                batch_size=100,
                num_workers=config['num_workers'],
                drop_last=False)

        return train_dataloader, val_dataloader

    def spilt_and_update11(dataloader, buffer, task_idx, config):

        print('using spilt_and_update11')
        
        train_dataset = deepcopy(dataloader.dataset)
        val_dataset = deepcopy(dataloader.dataset)

        # Train_loader
        images_train, images_val, labels_train, labels_val = slice_spilt(
            train_dataset.images,
            train_dataset.labels,
            test_size=0.1
        )

        train_dataset.images = train_dataset.images + buffer.train_images
        train_dataset.labels = train_dataset.labels + buffer.train_labels

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,  
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True)

        # Val_loader
        if task_idx == 0:
            val_dataloader = None
        else:
            current_num_per_classes_val = min(len(labels_val)//config['inc_cls_num'], (buffer.buffer_size * 0.1) // buffer.total_classes)
            print(f'Assigning {current_num_per_classes_val} per class in val data')

            current_num_per_classes_val = int(current_num_per_classes_val)

            selected_images_val, selected_labels_val = [], []

            for cls_label in np.unique(labels_val):
                
                cls_idx, = np.where(np.array(labels_val) == cls_label)
                cls_images, cls_labels = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

                selected_images_val.extend(cls_images[:current_num_per_classes_val])
                selected_labels_val.extend(cls_labels[:current_num_per_classes_val])

                print(f'{cls_label}, {len(cls_labels[:current_num_per_classes_val])}/{len(cls_labels)}')

            print(buffer.val_labels)

            for cls_label in range(buffer.total_classes):
                cls_idx, = np.where(np.array(buffer.val_labels) == cls_label)
                cls_images, cls_labels = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

                selected_images_val.extend(cls_images[:current_num_per_classes_val])
                selected_labels_val.extend(cls_labels[:current_num_per_classes_val])

                print(f'{cls_label}, {len(cls_labels[:current_num_per_classes_val])}/{len(cls_labels)}')

            val_dataset.images = selected_images_val
            val_dataset.labels = selected_labels_val

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=True,  
                batch_size=100,
                num_workers=config['num_workers'],
                drop_last=False)

        # Update Buffer
        buffer.total_classes += config['init_cls_num'] if task_idx == 0 else config['inc_cls_num']
        new_num_per_classes_train = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.9)
        new_num_per_classes_val = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.1)

        preserved_images_train, preserved_labels_train = [], []
        preserved_images_val, preserved_labels_val = [], []

        # Preserved old in buffer
        for old_cls_label in np.unique(buffer.train_labels):
            
            cls_idx = np.where(buffer.train_labels == old_cls_label)
            cls_images_train, cls_labels_train = np.array(buffer.train_images)[cls_idx], np.array(buffer.train_labels)[cls_idx]

            cls_idx = np.where(buffer.val_labels == old_cls_label)
            cls_images_val, cls_labels_val = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

        # Add new into buffer
        for new_cls_label in np.unique(labels_train):
            
            cls_idx = np.where(labels_train == new_cls_label)
            cls_images_train, cls_labels_train = np.array(images_train)[cls_idx], np.array(labels_train)[cls_idx]

            cls_idx = np.where(labels_val == new_cls_label)
            cls_images_val, cls_labels_val = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

        buffer.train_images = preserved_images_train
        buffer.train_labels = preserved_labels_train
        buffer.val_images = preserved_images_val
        buffer.val_labels = preserved_labels_val

        return train_dataloader, val_dataloader

    @staticmethod
    def spilt_and_update22(dataloader, buffer, task_idx, config):

        print('using spilt_and_update22')
        
        train_dataset = deepcopy(dataloader.dataset)
        val_dataset = deepcopy(dataloader.dataset)

        # Train_loader
        images_train, images_val, labels_train, labels_val = slice_spilt(
            train_dataset.images,
            train_dataset.labels,
            test_size=0.1
        )
        
        #train_dataset.images = buffer.train_images + train_dataset.images
        #train_dataset.labels = buffer.train_labels + train_dataset.labels

        train_dataset.images = train_dataset.images + buffer.train_images
        train_dataset.labels = train_dataset.labels + buffer.train_labels

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,  
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=True)

        # Update Buffer
        buffer.total_classes += config['init_cls_num'] if task_idx == 0 else config['inc_cls_num']
        new_num_per_classes_train = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.9)
        new_num_per_classes_val = int((config['buffer']['kwargs']['buffer_size'] // buffer.total_classes) * 0.1)

        preserved_images_train, preserved_labels_train = [], []
        preserved_images_val, preserved_labels_val = [], []

        # Preserved old in buffer
        for old_cls_label in np.unique(buffer.train_labels):
            
            cls_idx = np.where(buffer.train_labels == old_cls_label)
            cls_images_train, cls_labels_train = np.array(buffer.train_images)[cls_idx], np.array(buffer.train_labels)[cls_idx]

            cls_idx = np.where(buffer.val_labels == old_cls_label)
            cls_images_val, cls_labels_val = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

            print(f'{old_cls_label}, {len(cls_labels_val[:new_num_per_classes_val])}/{len(cls_labels_val)}')
        
        # Add new into buffer
        for new_cls_label in np.unique(labels_train):
            
            cls_idx = np.where(labels_train == new_cls_label)
            cls_images_train, cls_labels_train = np.array(images_train)[cls_idx], np.array(labels_train)[cls_idx]

            cls_idx = np.where(labels_val == new_cls_label)
            cls_images_val, cls_labels_val = np.array(images_val)[cls_idx], np.array(labels_val)[cls_idx]

            preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
            preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            preserved_images_val.extend(cls_images_val[:new_num_per_classes_val])
            preserved_labels_val.extend(cls_labels_val[:new_num_per_classes_val])

            print(f'{new_cls_label}, {len(cls_labels_val[:new_num_per_classes_val])}/{len(cls_labels_val)}')

        buffer.train_images = preserved_images_train
        buffer.train_labels = preserved_labels_train
        buffer.val_images = preserved_images_val
        buffer.val_labels = preserved_labels_val

        # Val loader
        if task_idx == 0:
            val_dataloader = None
        else:
            print(f'Assigning {new_num_per_classes_val} per class in val data')

            val_dataset.images = buffer.val_images
            val_dataset.labels = buffer.val_labels

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=True,  
                batch_size=100,
                num_workers=config['num_workers'],
                drop_last=False)

        return train_dataloader, val_dataloader

    @staticmethod
    def spilt_and_update4(dataloader, buffer, task_idx, config):

        print('using spilt_and_update4')

        buffer_size = config['buffer']['kwargs']['buffer_size']

        train_dataset = deepcopy(dataloader.dataset)
        val_dataset = deepcopy(dataloader.dataset)

        current_images = train_dataset.images
        current_labels = train_dataset.labels

        if task_idx == 0:

            buffer.total_classes += config['init_cls_num']
            new_num_per_classes_train = int(buffer_size * 0.9) // buffer.total_classes
            new_num_per_classes_val = int(buffer_size * 0.1) // buffer.total_classes

            ratio = (buffer_size * 0.1) / len(current_labels)

            images_train, images_val, labels_train, labels_val = balance_spilt(
                current_images,
                current_labels,
                test_size=ratio,
                random_state=config['seed']
            )

            # Some Assertions
            value_counts = Counter(labels_train)
            count1 = next(iter(value_counts.values()))
            for value, count in value_counts.items():
                assert count == count1

            value_counts = Counter(labels_val)
            count1 = next(iter(value_counts.values()))
            for value, count in value_counts.items():
                assert count == count1
        




            print(ratio, len(labels_train), len(labels_val))

            train_dataset.images = images_train
            train_dataset.labels = labels_train

            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,  
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                drop_last=True)

            buffer.val_images = images_val
            buffer.val_labels = labels_val

            preserved_images_train, preserved_labels_train = [], []
            for cls_label in range(config['init_cls_num']):
                
                cls_idx = np.where(labels_train == cls_label)
                cls_images_train, cls_labels_train = np.array(images_train)[cls_idx], np.array(labels_train)[cls_idx]
            
                preserved_images_train.extend(cls_images_train[:new_num_per_classes_train])
                preserved_labels_train.extend(cls_labels_train[:new_num_per_classes_train])

            buffer.train_images = preserved_images_train
            buffer.train_labels = preserved_labels_train

            val_dataloader = None

        else:

            buffer.total_classes += config['inc_cls_num']
            new_num_per_classes_train = int(buffer_size * 0.9) // buffer.total_classes
            new_num_per_classes_val = int(buffer_size * 0.1) // buffer.total_classes

            ratio = new_num_per_classes_val * config['inc_cls_num'] / len(current_labels)
            
            images_train, images_val, labels_train, labels_val = balance_spilt(
                current_images,
                current_labels,
                test_size=ratio,
                random_state=config['seed']
            )
            
            print(ratio, len(labels_train), len(labels_val))

            train_dataset.images = images_train + buffer.train_images
            train_dataset.labels = labels_train + buffer.train_labels

            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,  
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                drop_last=True)

            buffer.train_images.extend(images_train)
            buffer.train_labels.extend(labels_train)

            buffer.val_images.extend(images_val)
            buffer.val_labels.extend(labels_val)

            preserved_train_images, preserved_train_labels = [], []
            preserved_val_images, preserved_val_labels = [], []
            for cls_label in range(buffer.total_classes):

                cls_idx = np.where(np.array(buffer.train_labels) == cls_label)
                cls_train_images, cls_train_labels = np.array(buffer.train_images)[cls_idx], np.array(buffer.train_labels)[cls_idx]

                cls_idx = np.where(np.array(buffer.val_labels) == cls_label)
                cls_val_images, cls_val_labels = np.array(buffer.val_images)[cls_idx], np.array(buffer.val_labels)[cls_idx]

                preserved_train_images.extend(cls_train_images[:new_num_per_classes_train])
                preserved_train_labels.extend(cls_train_labels[:new_num_per_classes_train])

                preserved_val_images.extend(cls_val_images[:new_num_per_classes_val])
                preserved_val_labels.extend(cls_val_labels[:new_num_per_classes_val])

                
                print(f'{cls_label}, {len(cls_val_labels[:new_num_per_classes_val])}/{len(cls_val_labels)}')

            buffer.train_images = preserved_train_images
            buffer.train_labels = preserved_train_labels
            buffer.val_images = preserved_val_images
            buffer.val_labels = preserved_val_labels

            print(f'Assigning {new_num_per_classes_val} per class in val data')

            val_dataset.images = buffer.val_images
            val_dataset.labels = buffer.val_labels

            assert len(buffer.val_labels) == new_num_per_classes_val * buffer.total_classes

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=True,  
                batch_size=100,
                num_workers=config['num_workers'],
                drop_last=False)

        return train_dataloader, val_dataloader