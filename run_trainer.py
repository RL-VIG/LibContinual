import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer
import time

def main(rank, config):
    begin = time.time()
    trainer = Trainer(rank, config)
    trainer.train_loop()
    print("Time cost : ",time.time()-begin)

if __name__ == "__main__":
<<<<<<< HEAD
    config = Config("./config/lwf.yaml").get_config_dict()
=======
    config = Config("./config/der.yaml").get_config_dict()
>>>>>>> fee4c4b4e5801a983590fca3e84a2d4f27d67ddb

    if config["n_gpu"] > 1:
        pass
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        # torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
