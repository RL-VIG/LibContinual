import os
import torch
from torch import nn
from time import time
from tqdm import tqdm
from core.data import get_dataloader
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
import core.model as arch
from core.model.buffer import *
from torch.utils.data import DataLoader
import numpy as np


class Trainer(object):
    """
    The Trainer.
    
    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, config):
        self.rank = rank
        self.config = config
        self.config['rank'] = rank
        self.distribute = self.config['n_gpu'] > 1
        # (
        #     self.result_path, 
        #     self.log_path, 
        #     self.checkpoints_path, 
        #     self.viz_path
        # ) = self._init_files(config)                     # todo   add file manage
        self.logger = ""                               # todo   add logger
        self.device = self._init_device(config) 
        # self.writer = self._init_writer(self.viz_path)   # todo   add tensorboard
        
        print(self.config)

        self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)
        self.model = self._init_model(config)  # todo add parameter select
        (
            self.train_loader,
            self.test_loader,
        ) = self._init_dataloader(config)
        
        self.buffer = self._init_buffer(config)
        (
            self.init_epoch,
            self.inc_epoch,
            self.optimizer,
            self.scheduler,
        ) = self._init_optim(config)

        self.train_meter, self.test_meter = self._init_meter()

        self.val_per_epoch = config['val_per_epoch']

    def _init_device(self, config):
        """"
        Init the devices from the config.
        
        Args:
            config(dict): Parsed config file.
            
        Returns:
            device: a device.
        """
        init_seed(config['seed'], config['deterministic'])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_ids'])

        device = torch.device("cuda:{}".format(config['device_ids']))

        return device


    def _init_files(self, config):
        pass

    def _init_writer(self, config):
        pass

    def _init_meter(self, ):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
            # self.writer,
        )

        test_meter = [AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            # self.writer,
        ) for _ in range(self.task_num)]

        return train_meter, test_meter

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """
        params_dict_list = {"params": self.model.parameters()}
    
        optimizer = get_instance(
            torch.optim, "optimizer", config, params=self.model.parameters()
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details
        print(optimizer)
        

        if 'init_epoch' in config.keys():
            init_epoch = config['init_epoch']
        else:
            init_epoch = config['epoch']
        
        return init_epoch, config['epoch'], optimizer, scheduler

    def _init_data(self, config):
        return config['init_cls_num'], config['inc_cls_num'], config['task_num']

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        backbone = get_instance(arch, "backbone", config)
        dic = {"backbone": backbone}

        model = get_instance(arch, "classifier", config, **dic)
        print(model)
        print("Trainable params in the model: {}".format(count_parameters(model)))

        return model
    
    def _init_dataloader(self, config):
        '''
        Init DataLoader

        Args:
            config (dict): Parsed config file.

        Returns:
            train_loaders (list): Each task's train dataloader.
            test_loaders (list): Each task's test dataloader.
        '''
        train_loaders = get_dataloader(config, "train")
        test_loaders = get_dataloader(config, "test", cls_map=train_loaders.cls_map)

        return train_loaders, test_loaders
    
    def _init_buffer(self, config):
        '''
        Init Buffer
        
        Args:
            config (dict): Parsed config file.

        Returns:
            buffer (Buffer): a buffer for old samples.
        '''
        buffer = get_instance(arch, "buffer", config)

        return buffer

    def train_loop(self,):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        experiment_begin = time()
        for task_idx in range(self.task_num):
            print("================Task {} Start!================".format(task_idx))
            if hasattr(self.model, 'before_task'):
                self.model.before_task(task_idx)
            

            dataloader = self.train_loader.get_loader(task_idx)

            if isinstance(self.buffer, LinearBuffer) and task_idx != 0:
                datasets = dataloader.dataset
                datasets.images.extend(self.buffer.images)
                datasets.labels.extend(self.buffer.labels)
                dataloader = DataLoader(
                    datasets,
                    shuffle = True,
                    batch_size = 32,
                    drop_last = True
                )
            
            print("================Task {} Training!================".format(task_idx))
            print("The training samples number: {}".format(len(dataloader.dataset)))

            best_acc = 0.
            for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):
                print("learning rate: {}".format(self.scheduler.get_last_lr()))
                print("================ Train on the train set ================")
                train_acc = self._train(epoch_idx, dataloader)
                print(" * Average Acc: {:.3f} ".format(train_acc.avg("acc1")))

                if (epoch_idx+1) % self.val_per_epoch == 0:
                    print("================ Test on the test set ================")
                    test_acc = self._validate(task_idx)
                    best_acc = max(test_acc["avg_acc"], best_acc)
                    print(
                    " * Average Acc: {:.3f} Best acc {:.3f}".format(test_acc["avg_acc"], best_acc)
                    )
                    print(
                    " Per-Task Acc:{}".format(test_acc['per_task_acc'])
                    )
            
            self.buffer.update(self.train_loader.get_loader(task_idx).dataset, task_idx)



    def _train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        self.model.train()
        meter = self.train_meter
        meter.reset()
        

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.observe(batch)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                pbar.update(1)
                
                meter.update("acc1", acc)


        return meter



    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        meter = self.test_meter
        
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter[t].reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter[t].update("acc1", acc)

                per_task_acc.append(meter[t].avg("acc1"))
        
        return {"avg_acc" : np.mean(per_task_acc), "per_task_acc" : per_task_acc}
    


