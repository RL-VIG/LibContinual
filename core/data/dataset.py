from torch.utils.data import Dataset
import PIL
import numpy as np



class ContinualDatasets:
    def __init__(self, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms):
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.task_id = -1
        self.dataloaders = []

        self.create_loaders()

    def create_loaders(self):
        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + i * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(SingleDataseat(self.data_root, self.cls_map, start_idx, end_idx, self.trfms))

    def get_next_loader(self):
        self.task_id += 1
        assert self.task_id < self.task_num
        return self.dataloaders[self.task_id]

    
class SingleDataseat(Dataset):
    def __init__(self, data_root, cls_map, start_idx, end_idx, trfms):
        super().__init__()
        self.data_root = data_root
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms

        self.images = xxx
        self.labels = xxx


    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        image = self.trfms(image)
        return {"image": image, "label": label}
    

        