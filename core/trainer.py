import os
import torch
from torch import nn
from time import time
from core.data import get_dataloader
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
import core.model as arch

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
        self.train_meter, self.test_meter = self._init_meter()
        print(self.config)

        self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)        # todo  add continual setting
        self.model = self._init_model(config)
        (
            self.train_loader,
            self.test_loader
        ) = self._init_dataloader(config)

        (
            self.optimizer,
            self.scheduler,
        ) = self._init_optim(config)


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

        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            # self.writer,
        )

        return train_meter, test_meter

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """
        params_idx = []
        params_dict_list = []
        if "other" in config["optimizer"] and config["optimizer"]["other"] is not None:
            for key, value in config["optimizer"]["other"].items():
                # FIXME: Set the learning rate for the model specified parameter, including nn.Module and nn.Parameter
                # Original:
                # Pre-modified:
                # if self.distribute:
                #     sub_model = getattr(self.model.module, key)
                # else:
                #     sub_model = getattr(self.model, key)

                # Modified:
                sub_model = eval("self.model." + key)
                if isinstance(sub_model, nn.Module):
                    sub_model_params_list = list(sub_model.parameters())
                elif isinstance(sub_model, nn.Parameter):
                    sub_model_params_list = [sub_model]
                else:
                    raise Exception("Unrecognized type in optimizer.other")

                params_idx.extend(list(map(id, sub_model_params_list)))
                if value is None:
                    for p in sub_model_params_list:
                        p.requires_grad = False
                else:
                    param_dict = {"params": sub_model_params_list}
                    if isinstance(value, float):
                        param_dict.update({"lr": value})
                    elif isinstance(value, dict):
                        param_dict.update(value)
                    else:
                        raise Exception("Wrong config in optimizer.other")
                    params_dict_list.append(param_dict)

        params_dict_list.append(
            {
                "params": filter(
                    lambda p: id(p) not in params_idx, self.model.parameters()
                )
            }
        )
        optimizer = get_instance(
            torch.optim, "optimizer", config, params=params_dict_list
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details
        print(optimizer)
        

        return optimizer, scheduler

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

    def train_loop(self,):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        experiment_begin = time()
        for task_idx in range(self.task_num):
            print("================Task {} Start!================".format(task_idx))
            if hasattr(self.model, 'before_task'):
                self.model.before_task(task_idx)
            
            print("================Task {} Training!================".format(task_idx))
            for epoch_idx in range(self.epoch):
                print("learning rate: {}".format(self.scheduler.get_last_lr()))
                print("================ Train on the train set ================")
                train_acc = self._train(epoch_idx)
                print(" * Average Acc: {:.3f} ".format(train_acc['avg_acc']))

                if (epoch_idx+1) % self.eval_per_epoch == 0:
                    print("================ Test on the test set ================")
                    test_acc = self._validate(task_idx)
                    self.best_acc = max(test_acc['avg_acc'], self.best_acc)
                    print(
                    " * Average Acc: {:.3f} Best acc {:.3f}".format(test_acc['avg_acc'], self.best_val_acc)
                    )
                    print(
                    " Per-Task Acc:{}".format(test_acc['per_task_acc'])
                    )
                



    def _train(self, epoch_idx):
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

        for batch_idx, batch in enumerate(self.train_loader):
            output, acc, loss = self.model(batch)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            
            meter.update("avg_acc", acc)

        return meter



