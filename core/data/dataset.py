import os
from re import A
from typing import List
import PIL
import numpy as np

from torch.utils.data import Dataset, DataLoader

class ContinualDatasets:
    def __init__(self, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batchsize, num_workers):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.batchsize = batchsize
        self.num_workers = num_workers

        self.create_loaders()

    def create_loaders(self):
        self.dataloaders = []
        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(DataLoader(
                SingleDataset(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
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
            dataset = SingleDataset(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms)

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
                    # print(cls_idx)
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
    def __init__(self, data_root, mode, cls_map, start_idx, end_idx, trfms):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms

        self.images, self.labels, self.labels_name = self._init_datalist()

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = PIL.Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
        image = self.trfms(image)

        return {"image": image, "label": label}
    
    def __len__(self,):
        return len(self.labels)

    def _init_datalist(self):
        imgs, labels, labels_name = [], [], []
        for id in range(self.start_idx, self.end_idx):
            img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
            imgs.extend(img_list)
            labels.extend([id for _ in range(len(img_list))])
            labels_name.append(self.cls_map[id])
        
        return imgs, labels, labels_name

    def get_class_names(self):
        return self.labels_name


class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = self.images[index]
        image = PIL.Image.fromarray(np.uint8(image))
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        label = torch.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)


class Exemplar:
    def __init__(self, max_size, total_cls):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size
        self.total_classes = total_cls

    def update(self, cls_num, train, val):
        train_x, train_y = train
        val_x, val_y = val
        assert self.cur_cls == len(list(self.val.keys()))
        assert self.cur_cls == len(list(self.train.keys()))
        cur_keys = list(set(val_y))
        self.cur_cls += cls_num
        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)
        for key, value in self.val.items():
            self.val[key] = value[:val_store_num]
        for key, value in self.train.items():
            self.train[key] = value[:train_store_num]

        for x, y in zip(val_x, val_y):
            if y not in self.val:
                self.val[y] = [x]
            else:
                if len(self.val[y]) < val_store_num:
                    self.val[y].append(x)
        assert self.cur_cls == len(list(self.val.keys()))
        for key, value in self.val.items():
            assert len(self.val[key]) == val_store_num

        for x, y in zip(train_x, train_y):
            if y not in self.train:
                self.train[y] = [x]
            else:
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)
        assert self.cur_cls == len(list(self.train.keys()))
        for key, value in self.train.items():
            assert len(self.train[key]) == train_store_num

    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            for val_x in value:
                exemplar_val_x.append(val_x)
                exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls
