import sys

sys.dont_write_bytecode = True

import os
import re
import time
import torch
import argparse
import subprocess
import torch.multiprocessing as mp

from core.config import Config
from core import Trainer

def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Name of config file')
    args = parser.parse_args()

    if args.config:
        args.config = args.config + '.yaml' if not args.config.endswith('.yaml') else args.config
        config = Config(f'./config/{args.config}').get_config_dict()
    else:
        config = Config("./config/InfLoRA.yaml").get_config_dict()    

    if config['device_ids'] == 'auto':
        least_utilized_device = 0
        lowest_utilization = float('inf')

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi error: {result.stderr}")

            gpu_info = result.stdout.strip().split('\n')
            gpu_utilization = []

            for gpu in gpu_info:
                match = re.match(r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', gpu)
                if match:
                    device_id, mem_used, mem_total, gpu_util = map(int, match.groups())
                    # Combine memory usage and GPU utilization to determine the utilization score
                    utilization_score = gpu_util + (mem_used / mem_total) * 100
                    gpu_utilization.append((device_id, utilization_score))

            # Sort GPUs by utilization score (ascending) and select the least utilized GPUs
            gpu_utilization.sort(key=lambda x: x[1])
            config["device_ids"] = [str(gpu[0]) for gpu in gpu_utilization[:config["n_gpu"]]]

            print(f'Selected GPUs: {config["device_ids"]}')

        except Exception as e:
            config["device_ids"] = range(config["n_gpu"])
            print(f"Error while querying GPUs: {e}, using default device {config['device_ids']}")

    if not isinstance(config['device_ids'], list):
        config['device_ids'] = [config['device_ids']]

    if config["n_gpu"] > 1:
        mp.spawn(main, nprocs=config["n_gpu"], args=(config,))
        pass
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
    else:
        main(0, config)
