from torchvision import transforms
# from .augments import *
import os
import numpy as np
from .dataset import ContinualDatasets

# # MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
# # STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
# MEAN = [0.5071,  0.4866,  0.4409]
# STD = [0.2009,  0.1984,  0.2023]
# # WA
# # MEAN = [0.5071, 0.4867, 0.4408]
# # STD = [0.2675, 0.2565, 0.2761]
from .data import transform_classes


def get_augment(config, mode='train'):
    d = {'dataset': 'cifar', 
         'backbone': 'resnet',
         'mode': mode}
    
    if 'dataset' in config.keys():
        d['dataset'] = config['dataset']
    if 'vit' in config['backbone']['name'].lower():
        d['backbone'] = 'vit'
        
    return transform_classes[d['dataset']].get_transform(d['backbone'], d['mode'])
    

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

    trfms = get_augment(config, mode)
    # trfms = get_augment_method(config, mode)
    # trfms_list.append(transforms.ToTensor())
    # trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    # trfms = transforms.Compose(trfms_list)

    if cls_map is None:
        cls_list = os.listdir(os.path.join(data_root, mode))
        perm = np.random.permutation(len(cls_list))
        cls_map = dict()
        for label, ori_label in enumerate(perm):
            cls_map[label] = cls_list[ori_label]

    return ContinualDatasets(mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, config['batch_size'])




    
    