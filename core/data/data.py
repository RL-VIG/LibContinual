import numpy as np
from torchvision import transforms

class CIFARTransform:
    MEAN = [0.5071,  0.4866,  0.4409]
    STD = [0.2675, 0.2565, 0.2761]
    
    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN, std=STD)]
    
    resnet_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs
    ])
    
    resnet_test_transform = transforms.Compose([*common_trfs])
    
    # To Reproduce ERAML, ERACE
    #resnet_train_transform = transforms.Compose([*common_trfs])

    # from 
    dset_mean = (0., 0., 0.)
    dset_std = (1., 1., 1.)
    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)])
    
    vit_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)])

    # from trust region gradient projection
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    alexnet_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    alexnet_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

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
        elif model_type == 'alexnet':
            if mode == 'train':
                return CIFARTransform.alexnet_train_transform
            elif mode == 'test':
                return CIFARTransform.alexnet_test_transform
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
    
class ImageNetRTransform:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean, std)]
    
    resnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs])
    
    resnet_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        *common_trfs])

    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean, std)]

    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        *common_trfs])
    
    vit_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        *common_trfs])
    

    # from trust region gradient projection
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    alexnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    alexnet_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    @staticmethod
    def get_transform(model_type, mode):
        if model_type == 'resnet':
            if mode == 'train':
                return ImageNetRTransform.resnet_train_transform
            elif mode == 'test':
                return ImageNetRTransform.resnet_test_transform
        elif model_type == 'vit':
            if mode == 'train':
                return ImageNetRTransform.vit_train_transform
            elif mode == 'test':
                return ImageNetRTransform.vit_test_transform
        elif model_type == 'alexnet':
            if mode == 'train':
                return ImageNetRTransform.alexnet_train_transform
            elif mode == 'test':
                return ImageNetRTransform.alexnet_test_transform
        else:
            raise ValueError("Unsupported model type")

class TinyImageNetTransform:
    # Standard normalization values for Tiny-ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN, std=STD)]

    # ResNet Transforms
    resnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs
    ])

    resnet_test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        *common_trfs
    ])

    # ViT Transforms (Using dataset mean/std as [0,0,0] and [1,1,1] for compatibility)
    dset_mean = (0., 0., 0.)
    dset_std = (1., 1., 1.)

    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)
    ])

    vit_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)
    ])

    # from trust region gradient projection
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    alexnet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    alexnet_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    @staticmethod
    def get_transform(model_type, mode):
        if model_type == 'resnet':
            if mode == 'train':
                return TinyImageNetTransform.resnet_train_transform
            elif mode == 'test':
                return TinyImageNetTransform.resnet_test_transform
        elif model_type == 'vit':
            if mode == 'train':
                return TinyImageNetTransform.vit_train_transform
            elif mode == 'test':
                return TinyImageNetTransform.vit_test_transform
        elif model_type == 'alexnet':
            if mode == 'train':
                return TinyImageNetTransform.alexnet_train_transform
            elif mode == 'test':
                return TinyImageNetTransform.alexnet_test_transform
        else:
            raise ValueError("Unsupported model type")


class FiveDatasetsTransform:
    MEAN = [0.5071, 0.4866,  0.4409]
    STD = [0.2675, 0.2565, 0.2761]
    
    common_trfs = [transforms.ToTensor(),
                   transforms.Normalize(mean=MEAN, std=STD)]
    
    resnet_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        *common_trfs
    ])
    
    resnet_test_transform = transforms.Compose([
        transforms.Resize(32),
        *common_trfs
    ])

    # from 
    dset_mean = (0., 0., 0.)
    dset_std = (1., 1., 1.)
    vit_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)])
    
    vit_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(dset_mean, dset_std)])

    # from trust region gradient projection
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    alexnet_train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    alexnet_test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    @staticmethod
    def get_transform(model_type, mode):
        if model_type == 'resnet':
            if mode == 'train':
                return FiveDatasetsTransform.resnet_train_transform
            elif mode == 'test':
                return FiveDatasetsTransform.resnet_test_transform
        elif model_type == 'vit':
            if mode == 'train':
                return FiveDatasetsTransform.vit_train_transform
            elif mode == 'test':
                return FiveDatasetsTransform.vit_test_transform
        elif model_type == 'alexnet':
            if mode == 'train':
                return FiveDatasetsTransform.alexnet_train_transform
            elif mode == 'test':
                return FiveDatasetsTransform.alexnet_test_transform
        else:
            raise ValueError("Unsupported model type")

transform_classes = {
    'cifar': CIFARTransform,
    'imagenet': ImageNetTransform,
    'imagenet-r': ImageNetRTransform,
    'tiny-imagenet': TinyImageNetTransform,
    '5-datasets': FiveDatasetsTransform
}