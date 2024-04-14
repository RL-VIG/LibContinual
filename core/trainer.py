import os
import torch
from torch import nn
from time import time
from tqdm import tqdm
from core.data import get_dataloader
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
import core.model as arch
from core.model.buffer import *
from core.model import bic
from torch.utils.data import DataLoader
import numpy as np
import sys
from core.utils import Logger, fmt_date_str
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from copy import deepcopy
from pprint import pprint

from core.scheduler import CosineSchedule

class Trainer(object):
    """
    The Trainer.
    
    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, config):
        self.rank = rank
        self.config = config
        self.config['rank'] = rank
        self.distribute = self.config['n_gpu'] > 1  # 暂时不考虑分布式训练
        self.logger = self._init_logger(config)           
        self.device = self._init_device(config) 
        
        pprint(self.config)

        self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)
        self.model = self._init_model(config) 
        (
            self.train_loader,
            self.test_loader,
        ) = self._init_dataloader(config)
        
        self.buffer = self._init_buffer(config)

        self.task_idx = 0 
        (
            self.init_epoch,
            self.inc_epoch,
            self.optimizer,
            self.scheduler,
        ) = self._init_optim(config)

        self.train_meter, self.test_meter = self._init_meter()

        self.val_per_epoch = config['val_per_epoch']

        
        if self.config["classifier"]["name"] == "bic":
            self.stage2_epoch = config['stage2_epoch']


    def _init_logger(self, config, mode='train'):
        '''
        Init logger.

        Args:
            config (dict): Parsed config file.

        Returns:
            logger (Logger)
        '''

        save_path = config['save_path']
        log_path = os.path.join(save_path, "log")
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_prefix = config['classifier']['name'] + "-" + config['backbone']['name'] + "-" + f"epoch{config['epoch']}" #mode
        log_file = os.path.join(log_path, "{}-{}.log".format(log_prefix, fmt_date_str()))

        logger = Logger(log_file)

        # hack sys.stdout
        sys.stdout = logger

        return logger

    def _init_device(self, config):
        """"
        Init the devices from the config.
        
        Args:
            config(dict): Parsed config file.
            
        Returns:
            device: a device.
        """
        init_seed(config['seed'], config['deterministic'])
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
        )

        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
        )

        return train_meter, test_meter

    def _init_optim(self, config, stage2=False):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """
        if stage2:
            optimizer = get_instance(
                torch.optim, "optimizer", config, params=self.model.get_parameters(config, stage2=True)
            )
        else:
            optimizer = get_instance(
                torch.optim, "optimizer", config, params=self.model.get_parameters(config)
            )

        
        scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", config, optimizer=optimizer)


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
        dic = {"backbone": backbone, "device": self.device}

        model = get_instance(arch, "classifier", config, **dic)
        print(backbone)
        print("Trainable params in the model: {}".format(count_parameters(model)))

        model = model.to(self.device)
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
            self.task_idx = task_idx
            print("================Task {} Start!================".format(task_idx))
            self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
            if hasattr(self.model, 'before_task'):
                self.model.before_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))
            
            (
                _, __,
                self.optimizer,
                self.scheduler,
            ) = self._init_optim(self.config)
                
            dataloader = self.train_loader.get_loader(task_idx)

            if isinstance(self.buffer, (LinearBuffer, LinearHerdingBuffer)) and task_idx != 0:

                if self.config['classifier']['name'] == "bic":
                    dataloader, val_dataloader = bic.split_data(copy.deepcopy(dataloader), copy.deepcopy(self.buffer), self.config['batch_size'], task_idx)
                
                else:
                    datasets = dataloader.dataset
                    datasets.images.extend(self.buffer.images)
                    datasets.labels.extend(self.buffer.labels)
                    dataloader = DataLoader(
                        datasets,
                        shuffle = True,
                        batch_size = self.config['batch_size'],
                        drop_last = False,
                        num_workers = 8
                    )


            print("================Task {} Training!================".format(task_idx))
            print("The training samples number: {}".format(len(dataloader.dataset)))

            best_acc = 0.
            for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):

                print("learning rate: {}".format(self.scheduler.get_last_lr()))
                print("================ Train on the train set ================")
                train_meter = self._train(epoch_idx, dataloader)
                print("Epoch [{}/{}] |\tLoss: {:.4f} \tAverage Acc: {:.2f} ".format(epoch_idx, self.init_epoch if task_idx == 0 else self.inc_epoch, train_meter.avg('loss'), train_meter.avg("acc1")))

                if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1)==self.inc_epoch:
                    print("================ Test on the test set ================")
                    test_acc = self._validate(task_idx)
                    best_acc = max(test_acc["avg_acc"], best_acc)
                    print(
                    " * Average Acc: {:.2f} Best acc {:.2f}".format(test_acc["avg_acc"], best_acc)
                    )
                    print(
                    " * Per-Task Acc:{}".format(test_acc['per_task_acc'])
                    )
            
                self.scheduler.step()

            if hasattr(self.model, 'after_task'):
                self.model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))



            # stage_2  train
            if self.config["classifier"]["name"] == "bic" and task_idx != 0:
                self.model.backbone.eval()
                (_, __,
                    self.optimizer,
                    self.scheduler,
                ) = self._init_optim(self.config, stage2=True)
                
                scheduler = GradualWarmupScheduler(
                    self.model.bias_optimizer, self.config
                )

                print("================ Train on the train set (stage2)================")
                for epoch_idx in range(self.stage2_epoch):
                    print("learning rate: {}".format(self.scheduler.get_last_lr()))
                    print("================ Train on the train set ================")
                    train_meter = self.stage2_train(epoch_idx, val_dataloader)
                    print("Epoch [{}/{}] |\tLoss: {:.3f} \tAverage Acc: {:.3f} ".format(epoch_idx, self.stage2_epoch, train_meter.avg('loss'), train_meter.avg("acc1")))


                    if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1)==self.inc_epoch:
                        print("================ Test on the test set (stage2)================")
                        test_acc = self._validate(task_idx)
                        best_acc = max(test_acc["avg_acc"], best_acc)
                        print(
                        " * Average Acc: {:.3f} Best acc {:.3f}".format(test_acc["avg_acc"], best_acc)
                        )
                        print(
                        " * Per-Task Acc:{}".format(test_acc['per_task_acc'])
                        )
            
                    self.scheduler.step()

            if self.buffer.buffer_size > 0:
                if self.buffer.strategy == None:
                    pass
                if self.buffer.strategy == 'herding':
                    hearding_update(self.train_loader.get_loader(task_idx).dataset, self.buffer, self.model.backbone, self.device)
                elif self.buffer.strategy == 'random':
                    random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)
                

            print("================Task {} Testing!================".format(task_idx))
            test_acc = self._validate(task_idx)
            best_acc = max(test_acc["avg_acc"], best_acc)
            print(" * Average Acc: {:.2f} Best acc {:.2f}".format(test_acc["avg_acc"], best_acc))
            print(" * Per-Task Acc:{}".format(test_acc['per_task_acc']))
                    
    def stage2_train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        self.model.eval()
        for _ in range(len(self.model.bias_layers)):
            self.model.bias_layers[_].train()
        meter = self.train_meter
        meter.reset()
        

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.bias_observe(batch)

                #self.optimizer.zero_grad()

                #loss.backward()

                #self.optimizer.step()
                pbar.update(1)
                
                meter.update("acc1", acc)
                meter.update("loss", loss.item())


        return meter


    def _train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        self.model.train()
        meter = deepcopy(self.train_meter)
        meter.reset()

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.observe(batch)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                pbar.update(1)
                
                meter.update("acc1", 100 * acc)
                meter.update("loss", loss.item())


        return meter



    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        total_meter = deepcopy(self.test_meter)
        meter = deepcopy(self.test_meter)
        
        total_meter.reset()
        meter.reset()
        
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter.reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter.update("acc1", 100 * acc)
                    total_meter.update("acc1", 100 * acc)

                per_task_acc.append(round(meter.avg("acc1"), 2))
        
        return {"avg_acc" : round(total_meter.avg("acc1"), 2), "per_task_acc" : per_task_acc}
    


