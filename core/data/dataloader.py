from torchvision import transforms
from .augments import *
import os
import numpy as np
from .dataset import ContinualDatasets

MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

def get_dataloader(config, mode, cls_map=None):
    '''
    Initialize the dataloaders for Continual Learning.

    Args:
        config (dict): Parsed config dict.
        mode (string): 'trian' or 'test'.
        cls_map (dict): record the map between class and labels.
    
    Returns:
        Dataloaders (list): a list of dataloaders
    '''

    task_num = config['task_num']
    init_cls_num = config['init_cls_num']
    inc_cls_num = config['inc_cls_num']

    data_root = config['data_root']

    trfms_list = get_augment_method(config, mode)
    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    trfms = transforms.Compose(trfms_list)

    if cls_map is None:
        cls_list = os.listdir(os.path.join(data_root, mode))
        perm = np.random.permutation(len(cls_list))
        cls_map = dict()
        for label, ori_label in enumerate(perm):
            cls_map[label] = cls_list[ori_label]

    return ContinualDatasets(mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms)




