from .resnet import *
from .vit import vit_pt_imnet
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
        # SyntaxWarning: str indices must be integers or slices, not str; perhaps you missed a comma?
        # emb_func = eval(config["backbone"['name']])(**kwargs)
        emb_func = eval(config["backbone"]['name'])(**kwargs)
    except NameError:
        raise ("{} is not implemented".format(config["backbone"]['name']))
    
    return emb_func
