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
    parser.add_argument('--config', type=str, default=None, help='Name of config file')
    args = parser.parse_args()

    if args.config:
        args.config = args.config + '.yaml' if not args.config.endswith('.yaml') else args.config
        config = Config(f'./config/{args.config}').get_config_dict()
    else:
        config = Config("./config/InfLoRA.yaml").get_config_dict()
        
    if config["n_gpu"] > 1:
        pass
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    else:
        main(0, config)
