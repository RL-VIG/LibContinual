from .resnet import *
from .vit import vit_pt_imnet
from .SiNet import SiNet_vit
from .vit_inflora import Attention_LoRA
from .resnet_cbam import *

def get_backbone(config):
    """
    Get the backbone according to the config dict.

    Args:
        config: The config dict.

    Returns: The backbone module.
    """

    kwargs = dict()
    kwargs.update(config['backbone']['kwargs'])
    try:
        emb_func = eval(config["backbone"['name']])(**kwargs)
    except NameError:
        raise ("{} is not implemented".format(config["backbone"]['name']))
    
    return emb_func
