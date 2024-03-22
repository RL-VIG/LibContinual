import numpy as np
import torch
import torch.nn as nn
import PIL
import os
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class LinearHerdingBuffer:
    def __init__(self, buffer_size, batch_size):    
        self.buffer_size = buffer_size
        self.strategy = None
        self.batch_size = batch_size
        self.images, self.labels = [], []
        self.total_classes = 0

    def is_empty(self):
        return len(self.labels) == 0

    def clear(self):
        # clear the buffer
        del self.images
        del self.labels
        self.images = []
        self.labels = []
    
    def get_all_data(self):
        # return images and labels in the format of np.array
        return np.array(self.images), np.array(self.labels)
    
    def add_data(self, data:List[str], targets:List[str]):
        # add data and its labels to the buffer
        self.images.extend(data)
        self.labels.extend(targets)    


    def update(self, model:nn.Module, train_loader, val_transform, task_idx:int, 
               total_cls_num:int, cur_cls_indexes, device):
        
        # get the chosen global index in the dataset for buffer
        chosen_indexes = self.herding_select(model, train_loader, val_transform, 
                                             task_idx, total_cls_num, cur_cls_indexes, 
                                             device)
        
        cur_task_dataset = train_loader.dataset
        new_images = []
        new_labels = []
        for i in chosen_indexes:
            new_images.append(cur_task_dataset.images[i])
            new_labels.append(cur_task_dataset.labels[i])
            
        self.add_data(new_images, new_labels)


        
    def reduce_old_data(self, task_idx:int, total_cls_num:int) -> None:
        # subsample previous categories in the buffer
        samples_per_class = self.buffer_size // total_cls_num

        if task_idx > 0:
            buffer_X, buffer_Y = self.get_all_data()
            self.clear()
            for y in np.unique(buffer_Y):
                idx = (buffer_Y == y)
                selected_X, selected_Y = buffer_X[idx], buffer_Y[idx]
                self.add_data(
                    data=selected_X[:samples_per_class],
                    targets=selected_Y[:samples_per_class],
                )

    
    def herding_select(self, model:nn.Module, train_loader, val_transform, 
                       task_idx:int, total_cls_num:int, cur_cls_indexes, device):

        # Remove buffer samples from the dataset
        # and keep only the samples belonging to the current task category.
        def remove_buffer_sample_in_dataset(dataset, cur_cls_indexes):
            new_labels = []
            new_images = []
            for i in cur_cls_indexes:
                ind = np.array(dataset.labels) == i
                new_images.extend(list(np.array(dataset.images)[ind]))
                new_labels.extend(list(np.array(dataset.labels)[ind]))
            dataset.labels = new_labels
            dataset.images = new_images

        # get dataset containing buffer samples
        dataset = train_loader.dataset

        # remove buffer samples and only keep 
        remove_buffer_sample_in_dataset(dataset, cur_cls_indexes)

        # reset the transform
        dataset.trfms = val_transform

        # get loader for herding
        loader = DataLoader(
                dataset,
                # Note that `shuffle = False` should be set.
                # otherwise otherwise the generated indexes will not match with the paths of the images
                shuffle = False,
                batch_size = 32,
                # `drop_last = False` should be set as False, otherwise some samples are lost
                drop_last = False
            )
        
        # how many sample per class do we want
        samples_per_class = self.buffer_size // total_cls_num
    
        # compute feature for all training sample for all train samples
        extracted_features = []
        extracted_targets = []
        # print("!!!!! The origin code is\'feats = model.backbone(image)['features'] \', change to \'feats = model.extract_vector(image) \' by WA")
        with torch.no_grad():
            model.eval()
            for data in loader:
                image = data['image'].to(device)
                label = data['label'].to(device)
                # feats = model.extract_vector(image)
                feats = model.backbone(image)['features']
                feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.append(label)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = (torch.cat(extracted_targets)).cpu()

        result = []
        for curr_cls in np.unique(extracted_targets):
            
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            cls_feats = extracted_features[cls_ind]
            mean_feat = cls_feats.mean(0, keepdim=True)
            running_sum = torch.zeros_like(mean_feat)
            i = 0
            begin_index = cls_ind[0]
            while i < samples_per_class and i < cls_feats.shape[0]:
                cost = (mean_feat - (cls_feats + running_sum) / (i + 1)).norm(2, 1)

                # Notice that the initial offset should be added
                # since indexes we want are global in the dataset
                # hence we should guarantee indexes belonging to the same class 
                # should be continuous
                idx_min = cost.argmin().item()
                global_index = idx_min + begin_index
                result.append(global_index)
                running_sum += cls_feats[idx_min:idx_min + 1]
                cls_feats[idx_min] = cls_feats[idx_min] + 1e6
                i += 1

        return result
    

