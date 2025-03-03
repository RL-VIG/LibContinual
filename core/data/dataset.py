import os
import torch
import pickle
import numpy as np

from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

class ContinualDatasets:
    def __init__(self, dataset, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batchsize, num_workers, config):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.config = config
        self.dataset = dataset

        if self.dataset == 'binary_cifar100':
            datasets.CIFAR100(self.data_root, download = True)

        self.create_loaders()

    def create_loaders(self):
        self.dataloaders = []

        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(DataLoader(
                SingleDataset(self.dataset, self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
                shuffle = True,
                batch_size = self.batchsize,
                drop_last = False,
                num_workers = self.num_workers
            ))

    def get_loader(self, task_idx):
        assert task_idx >= 0 and task_idx < self.task_num
        if self.mode == 'train':
            return self.dataloaders[task_idx]
        else:
            return self.dataloaders[:task_idx+1]

class SingleDataset(Dataset):
    def __init__(self, dataset, data_root, mode, cls_map, start_idx, end_idx, trfms):
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        self.mode = mode
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms

        self.images, self.labels, self.labels_name = self._init_datalist()

    def __getitem__(self, idx):

        if self.dataset == 'binary_cifar100':

            image = self.images[idx]
            image = Image.fromarray(np.uint8(image))

        else:

            img_path = self.images[idx]
            image = Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
            
        label = self.labels[idx]
        image = self.trfms(image)

        return {"image": image, "label": label}
    
    def __len__(self,):
        return len(self.labels)

    def _init_datalist(self):

        imgs, labels, labels_name = [], [], []

        if self.dataset == 'binary_cifar100':
            
            with open(os.path.join(self.data_root, 'cifar-100-python', self.mode), 'rb') as f:
                load_data = pickle.load(f, encoding='latin1')

            for data, label in zip(load_data['data'], load_data['fine_labels']):

                if label in range(self.start_idx, self.end_idx):
                    r = data[:1024].reshape(32, 32)
                    g = data[1024:2048].reshape(32, 32)
                    b = data[2048:].reshape(32, 32)

                    tt_data = np.dstack((r, g, b))

                    imgs.append(tt_data)
                    labels.append(label)
                    labels_name.append(label)

        else:

            for id in range(self.start_idx, self.end_idx):
                img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
                imgs.extend(img_list)
                labels.extend([id for _ in range(len(img_list))])
                labels_name.append(self.cls_map[id])
            
        return imgs, labels, labels_name

    def get_class_names(self):
        return self.labels_name