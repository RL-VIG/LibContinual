from torch.utils.data import Dataset
import PIL
import numpy as np
import os
from torch.utils.data import DataLoader


class ContinualDatasets:
    def __init__(self, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batchsize):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.dataloaders = []
        self.batchsize = batchsize

        self.create_loaders()

    def create_loaders(self):
        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(DataLoader(
                SingleDataseat(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
                shuffle = True,
                batch_size = self.batchsize,
                drop_last = False
            ))

    def get_loader(self, task_idx):
        assert task_idx >= 0 and task_idx < self.task_num
        if self.mode == 'train':
            return self.dataloaders[task_idx]
        else:
            return self.dataloaders[:task_idx+1]
        


    
class SingleDataseat(Dataset):
    def __init__(self, data_root, mode, cls_map, start_idx, end_idx, trfms):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms

        self.images, self.labels = self._init_datalist()

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = PIL.Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
        image = self.trfms(image)
        return {"image": image, "label": label}
    
    def __len__(self,):
        return len(self.labels)

    def _init_datalist(self):
        imgs, labels = [], []
        for id in range(self.start_idx, self.end_idx):
            # print(id, self.cls_map[id])
            img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
            imgs.extend(img_list)
            labels.extend([id for _ in range(len(img_list))])
        
        return imgs, labels
        # return np.array(imgs), np.array(labels)

    

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