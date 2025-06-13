import os
import random
import numpy as np
import core.data.custom_transforms as cstf

from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .dataset import ContinualDatasets, ImbalancedDatasets
from .data import transform_classes
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

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
    # Special judge for RAPF
    if 'is_rapf' in config.keys() and config['is_rapf']:
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        n_px = config['image_size']

        return Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    if f'{mode}_trfms' in config.keys():
        return _create_transforms(config[f'{mode}_trfms'])

    # TODO: currently keeping below part for backward compatibility, will be remove in future

    d = {'dataset': 'cifar', 
         'backbone': 'resnet',
         'mode': mode}
    
    if 'dataset' in config.keys():
        if 'cifar' in config['dataset']:
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
    dataset = config['dataset']

    trfms = get_augment(config, mode)

    if f'{mode}_batch_size' in config.keys():
        batch_size = config[f'{mode}_batch_size']
    else:
        batch_size = config['batch_size']

    if dataset == 'tiny-imagenet':
        cls_map = {}
        with open(os.path.join(os.getcwd(), "core", "data", "dataset_reqs", f"tinyimagenet_classes.txt"), "r") as f:
            for line in f.readlines():
                _, cls_code, cls_name = line.strip().split('\t')
                cls_map[cls_code] = cls_name

    elif cls_map is None and dataset != 'binary_cifar100':
        # Apply class_order for debugging
        cls_list = sorted(os.listdir(os.path.join(data_root, mode)))
        #random.shuffle(cls_list)
        if 'class_order' in config.keys():
            class_order = config['class_order']
            perm = class_order
        else: 
            perm = np.random.permutation(len(cls_list))
        cls_map = dict()
        for label, ori_label in enumerate(perm):
            cls_map[label] = cls_list[ori_label]

    if mode == 'train' and 'imb_type' in config.keys():
        # generate long-tailed data to reproduce DAP
        return ImbalancedDatasets(mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batch_size, num_workers, config['imb_type'], config['imb_factor'], config['shuffle'])

    return ContinualDatasets(dataset, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms, batch_size, num_workers, config)
    
