from .resnet import *
from .vit import vit_pt_imnet
from .vit import vit_pt_imnet_in21k_adapter
from .vit_dap import vit_pt_imnet_dap

from .SiNet import SiNet_vit

from .resnet_cbam import *
from .alexnet import AlexNet_TRGP, AlexNet_API
from .clip import clip

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
        emb_func = eval(config["backbone"]['name'])(**kwargs)
    except NameError:
        raise ("{} is not implemented".format(config["backbone"]['name']))
    
    return emb_func
