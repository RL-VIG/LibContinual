from torchvision import transforms
from .autoaugment import *
from .cutout import *
from .randaugment import *

CJ_DICT = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}


def get_augment_method(
    config,
    mode,
):
    """Return the corresponding augmentation method according to the setting.

    + Use `ColorJitter` and `RandomHorizontalFlip` when not setting `augment_method` or using `NormalAug`.
    + Use `ImageNetPolicy()`when using `AutoAugment`.
    + Use `Cutout()`when using `Cutout`.
    + Use `RandAugment()`when using `RandAugment`.
    + Use `CenterCrop` and `RandomHorizontalFlip` when using `AutoAugment`.
    + Users can add their own augment method in this function.

    Args:
        config (dict): A LFS setting dict
        mode (str): mode in train/test/val

    Returns:
        list: A list of specific transforms.
    """

    # return []



    if mode == 'test':
        trfms_list = []
    else:
        trfms_list = []
        trfms_list.append(transforms.RandomCrop(32, padding=4))
        trfms_list.append(transforms.RandomHorizontalFlip())
        trfms_list.append(transforms.ColorJitter(brightness=63 / 255))
    return trfms_list

def get_default_image_size_trfms(image_size):
    """ Return the uniform transforms for image_size """
    if image_size == 224:
        trfms = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
        ]
    elif image_size == 84:
        trfms = [
            transforms.Resize((96, 96)),
            transforms.RandomCrop((84, 84)),
        ]
    # for MTL -> alternative solution: use avgpool(ks=11)
    elif image_size == 80:
        # MTL use another MEAN and STD
        trfms = [
            transforms.Resize((92, 92)),
            transforms.RandomResizedCrop(88),
            transforms.CenterCrop((80, 80)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise RuntimeError
    return trfms