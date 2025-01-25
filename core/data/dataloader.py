import os
import numpy as np
import core.data.custom_transforms as cstf

from torchvision import datasets, transforms
from .dataset import ContinualDatasets
from .data import transform_classes

def _create_transforms(cfg):
    transform_list = []

    for item in cfg:
        for func_name, params in item.items():
        
            # Convert str to enum, if required
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        params[k] = transforms.InterpolationMode[v]
                    except KeyError:
                        pass

            if func_name in cstf.custom_trfm_names:
                transform = getattr(cstf, func_name)
            else:
                transform = getattr(transforms, func_name)(**params)

            transform_list.append(transform)

    return transforms.Compose(transform_list)

def get_augment(config, mode='train'):

    if f'{mode}_trfms' in config.keys():
        return _create_transforms(config[f'{mode}_trfms'])

    # TODO: currently keeping below part for backward compatibility, will be remove in future

    d = {'dataset': 'cifar', 
         'backbone': 'resnet',
         'mode': mode}
    
    if 'dataset' in config.keys():
        if config['dataset'] == 'cifar100':
            d['dataset'] = 'cifar'
        else:
            d['dataset'] = config['dataset']
    if 'vit' in config['backbone']['name'].lower():
        d['backbone'] = 'vit'
    if 'alexnet' in config['backbone']['name'].lower():
        d['backbone'] = 'alexnet'
        
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
    num_workers = config['num_workers']

    trfms = get_augment(config, mode)

    if f'{mode}_batch_size' in config.keys():
        batch_size = config[f'{mode}_batch_size']
    else:
        batch_size = config['batch_size']

    if cls_map is None:
        cls_list = os.listdir(os.path.join(data_root, mode))
        perm = np.random.permutation(len(cls_list))
        cls_map = dict()
        for label, ori_label in enumerate(perm):
            cls_map[label] = cls_list[ori_label]

    return ContinualDatasets(mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batch_size, num_workers)
    