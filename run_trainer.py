import sys

sys.dont_write_bytecode = True

import argparse
import torch
import os
import time
from core.config import Config
from core import Trainer


def main(rank, config):
    begin = time.time()
    trainer = Trainer(rank, config)
    trainer.train_loop()
    print("Time cost : ",time.time()-begin)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_name', type=str, default=None, help='Name of config file')

    args = parser.parse_args()

    if args.config_name:
        config = Config(f'./config/{args.config_name}').get_config_dict()
    else:
        # config = Config("./config/der.yaml").get_config_dict()
        # config = Config("./config/eraml.yaml").get_config_dict()
        # config = Config("./config/erace.yaml").get_config_dict()
        # config = Config("./config/icarl.yaml").get_config_dict()
        # config = Config("./config/InfLoRA.yaml").get_config_dict()
        # config = Config("./config/trgp.yaml").get_config_dict()  
        config = Config("./config/InfLoRA_opt.yaml").get_config_dict()  
        config = Config("./config/InfLoRA_trgp.yaml").get_config_dict()  
        
    if config["n_gpu"] > 1:
        pass
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    else:
        main(0, config)
