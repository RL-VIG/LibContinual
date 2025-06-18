import argparse
import os
import random
import yaml
import re

def get_cur_path():
    """Get the absolute path of current file.

    Returns: The absolute path of this file (Config.py).

    """
    return os.path.dirname(__file__)


DEFAULT_FILE = os.path.join(get_cur_path(), "default.yaml")

class Config(object):
    """ The config parser of `LibContinual`
    `Config` is used to parser *.yaml, console params to python dict. The rules for resolving merge conflicts are as follow

    1. The merging is recursive, if a key is not be specified, the existing value will be used.
    2. The merge priority is: console params > run_*.py > user defined yaml (/LibContinual/config/*.yaml) > default.yaml(/LibContinual/core/config/*.yaml)
    """

    def __init__(self, config_file=None):
        """Initializing the parameter dictionary, completes the merging of all parameter.
        
        Args:
            config_file: Configuration file name. (/LibContinual/config/*.yaml)
        """
        self.config_file = config_file
        self.default_dict = self._load_config_files(DEFAULT_FILE)
        self.file_dict = self._load_config_files(config_file)
        self.console_dict = self._load_console_dict()
        self.config_dict = self._merge_config_dict()

    def get_config_dict(self):
        """ Return the merged dict.

        Returns:
            dict: A dict of LibContinual setting.
        """
        return self.config_dict


    @staticmethod
    def _load_config_files(config_file):
        """Parse a YAML file.
        
        Args:
            config_file (str): Path to yaml file.
    
        Returns:
            dict: A dict of LibContinual setting.
        """
        config_dict = dict()
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        config_file_dict = config_dict.copy()
        for include in config_dict.get("includes", []):
            with open(os.path.join("./config/", include), "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        if config_dict.get("includes") is not None:
            config_dict.pop("includes")
        config_dict.update(config_file_dict)
        return config_dict

    @staticmethod
    def _load_console_dict():
        """Parsing command line parameters
        
        Returns:
            dict: A dict of LibContinual console setting.
        """
        pass

    @staticmethod
    def _update(dic1, dic2):
        """Merge dictionaries.

        Used to merge two dictionaries (profiles), `dic2` will overwrite the value of the same key in `dic1`.

        
        Args:
            dic1 (dict): The dict to be overwritten. (low priority)
            dic2 (dict): The dict to overwrite. (high priority)

        Returns:
            dict: Merged dict.
        """

        if dic1 is None:
            dic1 = dict()

        if dic2 is not None:
            for k in dic2.keys():
                dic1[k] = dic2[k]
        return dic1


    def _merge_config_dict(self):
        """Merge all dictionaries. Merge rules are as follow

        1. The merging is recursive, if a key is not be specified, the existing value will be used.
        2. The merge priority is: console params > run_*.py > user defined yaml (/LibContinual/config/*.yaml) > default.yaml(/LibContinual/core/config/*.yaml)
        
        Returns:
            dict: A complete dict of LibContinual setting.
        """

        config_dict = dict()
        config_dict = self._update(config_dict, self.default_dict)
        config_dict = self._update(config_dict, self.file_dict)
        config_dict = self._update(config_dict, self.console_dict)

        return config_dict