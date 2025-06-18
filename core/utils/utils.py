import pandas as pd
import os
import torch
from datetime import datetime
import numpy as np
import random
from time import time
from torch.optim.lr_scheduler import _LRScheduler


class AverageMeter(object):
    """
    A AverageMeter to meter avg of number-like data.
    """

    def __init__(self, name, keys, writer=None):
        self.name = name
        self._data = pd.DataFrame(
            index=keys, columns=["last_value", "total", "counts", "average"]
        )
        self.writer = writer
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            tag = "{}/{}".format(self.name, key)
            self.writer.add_scalar(tag, value)
        # self._data.last_value[key] = value
        # self._data.total[key] += value * n
        # self._data.counts[key] += n
        # self._data.average[key] = self._data.total[key] / self._data.counts[key]
        self._data.loc[key, 'last_value'] = value
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']


    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def last(self, key):
        return self._data.last_value[key]

    def total(self, key):
        return self._data.total[key]
    


def init_seed(seed=0, deterministic=False):
    """

    :param seed:
    :param deterministic:
    :return:
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_instance(module, name, config, **kwargs):
    """
    A reflection function to get backbone/classifier/.

    Args:
        module ([type]): Package Name.
        name (str): Top level value in config dict. (backbone, classifier, etc.)
        config (dict): The parsed config dict.

    Returns:
         Corresponding instance.
    """
    if config[name]["kwargs"] is not None:
        kwargs.update(config[name]["kwargs"])
    
    return getattr(module, config[name]["name"])(**kwargs)

# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, config):
        # if self.multiplier < 1.:
        #     raise ValueError('multiplier should be greater thant or equal to 1.')
        self.optimizer = optimizer
        self.total_epoch = config["epoch"]
        self.warmup = config["warmup"]
        self.after_scheduler = self.get_after_scheduler(config)
        self.finish_warmup = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_after_scheduler(self, config):
        scheduler_name = config["lr_scheduler"]["name"]
        scheduler_dict = config["lr_scheduler"]["kwargs"]

        if self.warmup != 0:
            if scheduler_name == "CosineAnnealingLR":
                scheduler_dict["T_max"] -= self.warmup - 1
            elif scheduler_name == "MultiStepLR":
                scheduler_dict["milestones"] = [
                    step - self.warmup + 1 for step in scheduler_dict["milestones"]
                ]

        if scheduler_name == "LambdaLR":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=eval(config["lr_scheduler"]["kwargs"]["lr_lambda"]),
                last_epoch=-1,
            )

        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer=self.optimizer, **scheduler_dict
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup - 1:
            self.finish_warmup = True
            return self.after_scheduler.get_last_lr()

        return [
            base_lr * float(self.last_epoch + 1) / self.warmup
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
            if self.finish_warmup and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):

    return sum(p.numel() for p in model.parameters())

def fmt_date_str(date=None, fmt="%y-%m-%d-%H-%M-%S"):
    """Format date to string.

    Args:
        datetime (datetime, optional): get current time if None. Defaults to None.

    Returns:
        str: formatted date string
    """
    if date is None:
        date = datetime.now()
    return date.strftime(fmt)

def compute_bwt(acc_table, curr_acc, task_idx):
    '''
    After training T tasks, $BWT = \frac{\sum_{i=3}^T\sum_{j=1}^{i-2}R_{i,j}-R{j,j}}{T(T-1)/2}$
    Equivalent to Positive BwT of Continuum
    https://arxiv.org/pdf/1810.13166
    '''

    if task_idx > 1:
    
        bwt = 0.
        for i in range(2, task_idx):
            for j in range(i - 1):
                bwt += acc_table[i, j] - acc_table[j, j]

        for j in range(task_idx - 1):
            bwt += curr_acc[j] - acc_table[j, j]

        return (bwt * 2) / (task_idx * (task_idx+1))

    return 0.


def compute_frgt(acc_table, curr_acc, task_idx):
    '''
    After training T tasks, $Frgt = \frac{\sum_{j=1}^{T-2}R_{T-1,j}-R_{j,j}}{T-1}$
    Equivalent to Forgetting of Continuum
    '''

    if task_idx > 1:
        return sum(np.diag(acc_table)[:task_idx - 1] - curr_acc[:task_idx+1][:-2]) / task_idx
    return 0.


def compute_fps(model, config):
    model.eval()

    # assert image_size in config is Correct, check ranpac

    data = {
        'image' : torch.rand((2, 3, config['image_size'], config['image_size'])).cuda(),
        'label' : torch.zeros((2))
    }

    t_all = []

    for i in range(100):
        t1 = time()
        if config['setting'] == 'task-aware':
            model.inference(data, task_id = random.randint(0, config['task_num'] - 1))
        elif config['setting'] == 'task-agnostic':
            model.inference(data)
        t2 = time()
        t_all.append(t2 - t1)

    return {'avg_fps' : 1 / np.mean(t_all),
            'best_fps' : 1 / min(t_all)}