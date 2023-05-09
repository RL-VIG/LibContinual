from torch.utils.data import Dataset
import PIL
import numpy as np
import os
from torch.utils.data import DataLoader


class ContinualDatasets:
    def __init__(self, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.dataloaders = []

        self.create_loaders()

    def create_loaders(self):
        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(DataLoader(
                SingleDataseat(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
                shuffle = True,
                batch_size = 32,
                drop_last = True
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
            img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
            imgs.extend(img_list)
            labels.extend([id for _ in range(len(img_list))])

        return imgs, labels

    
