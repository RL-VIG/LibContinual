import os
import torch
import pickle
import random
import numpy as np

from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from continuum.datasets import TinyImageNet200
from continuum import ClassIncremental

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

        if self.dataset == 'tiny-imagenet':

            if 'class_order' in self.config:
                class_order = self.config['class_order']
            else:
                class_order = list(range(200))
                random.seed(self.config['seed'])
                random.shuffle(class_order)

            scenario = ClassIncremental(
                TinyImageNet200(self.data_root, train=self.mode == 'train', download=True),
                initial_increment=self.init_cls_num,
                increment=self.inc_cls_num,
                class_order=class_order
            )

            class_ids_per_task = (
                [class_order[:self.init_cls_num]] + 
                [class_order[i:i + self.inc_cls_num] for i in range(self.init_cls_num, len(class_order), self.inc_cls_num)]
            )

            with open(os.path.join(os.getcwd(), "core", "data", "dataset_reqs", f"tinyimagenet_classes.txt"), "r") as f:
                lines = f.read().splitlines()
            classes_names = [line.split("\t")[-1] for line in lines]

            for t in range(self.task_num):

                cur_scenario = scenario[t:t+1]

                dataset = SingleDataset(self.dataset, self.data_root, self.mode, self.init_cls_num, self.inc_cls_num, self.cls_map, self.trfms, init=False)
                dataset.images = cur_scenario._x
                dataset.labels = cur_scenario._y
                dataset.labels_name = [classes_names[class_id] for class_id in class_ids_per_task[t]]

                self.dataloaders.append(DataLoader(
                    dataset,
                    shuffle = True,
                    batch_size = self.batchsize,
                    drop_last = False,
                    num_workers = self.num_workers,
                    pin_memory=self.config['pin_memory']
                ))

        else:

            for i in range(self.task_num):

                start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
                end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
                self.dataloaders.append(DataLoader(
                    SingleDataset(self.dataset, self.data_root, self.mode, self.init_cls_num, self.inc_cls_num, self.cls_map, self.trfms, start_idx, end_idx),
                    shuffle = True,
                    batch_size = self.batchsize,
                    drop_last = False,
                    num_workers = self.num_workers,
                    pin_memory=False
                ))

    def get_loader(self, task_idx):
        assert task_idx >= 0 and task_idx < self.task_num
        if self.mode == 'train':
            return self.dataloaders[task_idx]
        else:
            return self.dataloaders[:task_idx+1]

class ImbalancedDatasets(ContinualDatasets):
    def __init__(self, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batchsize, num_workers, imb_type='exp', imb_factor=0.002, shuffle=False):
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.shuffle = shuffle
        super().__init__(mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batchsize, num_workers)

    def create_loaders(self):
        self.dataloaders = []
        cls_num = self.init_cls_num + self.inc_cls_num * (self.task_num - 1)
        img_num_list = self._get_img_num_per_cls(cls_num, self.imb_type, self.imb_factor)

        if self.shuffle:
            grouped_img_nums = [img_num_list[i:i + self.inc_cls_num] for i in range(0, cls_num, self.inc_cls_num)]
            np.random.shuffle(grouped_img_nums)
            for group in grouped_img_nums:
                np.random.shuffle(group)
            shuffled_img_num_list = [num for group in grouped_img_nums for num in group]
            img_num_list = shuffled_img_num_list

        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i - 1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i == 0 else self.inc_cls_num)
            dataset = SingleDataset(self.data_root, self.mode, self.cls_map, self.trfms, start_idx, end_idx)

            new_imgs, new_labels = [], []
            labels_np = np.array(dataset.labels, dtype=np.int64)
            classes = np.unique(labels_np)
            for the_class, the_img_num in zip(classes, img_num_list[i * self.inc_cls_num:(i + 1) * self.inc_cls_num]):
              idx = np.nonzero(labels_np == the_class)[0]
              np.random.shuffle(idx)
              selec_idx = idx[:the_img_num]
              new_imgs.extend([dataset.images[j] for j in selec_idx])
              new_labels.extend([the_class, ] * the_img_num)
            dataset.images = new_imgs
            dataset.labels = new_labels

            self.dataloaders.append(DataLoader(
                dataset,
                batch_size = self.batchsize,
                drop_last = False
            ))

    def _get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[0])))
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(max(int(num), 1))
        elif imb_type == 'exp_re':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(max(int(num), 1))
            img_num_per_cls.reverse()
        elif imb_type == 'exp_max':
            cls_per_group = cls_num//self.task_num
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'exp_max_re':
            cls_per_group = cls_num//self.task_num
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    # print(cls_idx)
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
            img_num_per_cls.reverse()

        elif imb_type == 'exp_min':
            cls_per_group = cls_num//self.task_num
            for cls_idx in range(cls_num):
                if (cls_idx+1)%cls_per_group==1:
                    # print(cls_idx)
                    num = img_max * (imb_factor**((cls_idx+cls_per_group-1) / (cls_num - 1.0)))
                    # print(num)
                img_num_per_cls.append(int(num))

        elif imb_type == 'half':
            cls_per_group = cls_num // self.task_num
            ratio = 2
            num = 1
            for cls_idx in range(cls_num):
                if num > img_max:
                    num = img_max
                img_num_per_cls.append(int(num))
                if (cls_idx + 1) % cls_per_group == 0:
                    num *= ratio
            img_num_per_cls.reverse()

        elif imb_type == 'half_re':
            cls_per_group = cls_num // self.task_num
            ratio = 2
            num = 1
            for cls_idx in range(cls_num):
                if num > img_max:
                    num = img_max
                img_num_per_cls.append(int(num))
                if (cls_idx + 1) % cls_per_group == 0:
                    num *= ratio

        elif imb_type == 'halfbal':
            cls_per_group = cls_num // self.task_num
            N = img_max * cls_per_group

            total = 0
            for i in range(self.task_num):
                total += N / (2**i)
            print(total)
            per_class_count = int(total / cls_num)
            img_num_per_cls.extend([per_class_count] * cls_num)

        elif imb_type == 'oneshot':
            img_num_per_cls.extend([1] * cls_num)
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == 'fewshot':
            for cls_idx in range(cls_num):
                if cls_idx<50:
                    num = img_max
                else:
                    num = img_max*0.01
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

class SingleDataset(Dataset):
    def __init__(self, dataset, data_root, mode, init_cls_num, inc_cls_num, cls_map, trfms, start_idx=-1, end_idx=-1, init=True):
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root
        self.mode = mode
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms

        if init:
            self.images, self.labels, self.labels_name = self._init_datalist()

    def __getitem__(self, idx):
        if self.dataset == 'binary_cifar100':

            image = self.images[idx]
            image = Image.fromarray(np.uint8(image))

        elif self.dataset == 'tiny-imagenet':
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")

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