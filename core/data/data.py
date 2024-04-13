import numpy as np
from torchvision import transforms

class CIFARTransform:
    MEAN = [0.5071,  0.4866,  0.4409]
    STD = [0.2009,  0.1984,  0.2023]
    
    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN, std=STD)]
    
    resnet_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs
    ])
    
    resnet_test_transform = transforms.Compose([*common_trfs])
    
    # from 
    dset_mean = (0., 0., 0.)
    dset_std = (1., 1., 1.)
    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std),
    ])
    
    vit_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std),
    ])
    
    @staticmethod
    def get_transform(model_type, mode):
        if model_type == 'resnet':
            if mode == 'train':
                return CIFARTransform.resnet_train_transform
            elif mode == 'test':
                return CIFARTransform.resnet_test_transform
        elif model_type == 'vit':
            if mode == 'train':
                return CIFARTransform.vit_train_transform
            elif mode == 'test':
                return CIFARTransform.vit_test_transform
        else:
            raise ValueError("Unsupported model type")
    
    
class ImageNetTransform:
    MEAN=[0.4914, 0.4822, 0.4465]
    STD=[0.2023, 0.1994, 0.2010]
    
    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN, std=STD)]
    
    resnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs
    ])
    
    resnet_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        *common_trfs
    ])
    
    
    dset_mean = (0., 0., 0.)
    dset_std = (1., 1., 1.)
    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std),
    ])
    
    vit_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std),
    ])
    
    @staticmethod
    def get_transform(model_type, mode):
        if model_type == 'resnet':
            if mode == 'train':
                return ImageNetTransform.resnet_train_transform
            elif mode == 'test':
                return ImageNetTransform.resnet_test_transform
        elif model_type == 'vit':
            if mode == 'train':
                return ImageNetTransform.vit_train_transform
            elif mode == 'test':
                return ImageNetTransform.vit_test_transform
        else:
            raise ValueError("Unsupported model type")
    
    
transform_classes = {
    'cifar': CIFARTransform,
    'imagenet': ImageNetTransform
}