import os
import sys
import torch

import numpy as np
import core.model as arch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from pprint import pprint
from contextlib import redirect_stdout
from time import time
from tqdm import tqdm
from core.data import get_dataloader
from core.utils import *
from core.model.buffer import *
from core.model import bic
from torch.utils.data import DataLoader
from core.utils import Logger, fmt_date_str
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from copy import deepcopy

from core.scheduler import CosineSchedule, PatienceSchedule, CosineAnnealingWarmUp

class Trainer(object):
    """
    The Trainer.
    
    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, config):

        self.rank = rank
        self.config = config
        self.distribute = self.config['n_gpu'] > 1  # 暂时不考虑分布式训练
        assert not self.distribute
        if self.distribute:
            dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=self.config['n_gpu'], rank=rank)
        self.logger = self._init_logger(config)           
        self.device = self._init_device(config)

        #pprint(config)
        # Write config into log file only
        with redirect_stdout(self.logger.file):
            pprint(config)
        
        
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

        log_path = os.path.join(save_path, "log", config['classifier']['name'])
        os.makedirs(log_path, exist_ok=True)
        
        log_prefix = f"{config['dataset']}..{config['backbone']['name']}--ep{config['epoch']}--s{config['seed']}__{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        log_file = os.path.join(log_path, f"{log_prefix}.log")
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

        device = torch.device(f'cuda:{config["device_ids"][self.rank]}')
        torch.cuda.set_device(device)

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

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """

        if 'init_epoch' in config.keys():
            init_epoch = config['init_epoch']
        else:
            init_epoch = config['epoch']

        model = self.model.module if self.distribute else self.model

        if self.task_idx == 0 and 'init_optimizer' in config.keys():
            optimizer = get_instance(
                torch.optim, "init_optimizer", config, params=model.get_parameters(config)
            )
        else:
            optimizer = get_instance(
                torch.optim, "optimizer", config, params=model.get_parameters(config)
            )

        # Check if the learning rate scheduler specified in the configuration is "CosineSchedule"
        if config['lr_scheduler']['name'] == "CosineSchedule":
            scheduler = CosineSchedule(optimizer, K=config['lr_scheduler']['kwargs']['K'])
        elif config['lr_scheduler']['name'] == "PatienceSchedule":
            scheduler = PatienceSchedule(optimizer, patience = config['lr_scheduler']['kwargs']['patience'], factor = config['lr_scheduler']['kwargs']['factor'])
        elif config['lr_scheduler']['name'] == "Constant":
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1)
        elif config['lr_scheduler']['name'] == "CosineAnnealingWarmUp":
            T_max = len(self.train_loader.get_loader(self.task_idx))
            T_max *= init_epoch if self.task_idx == 0 else config['epoch']
            scheduler = CosineAnnealingWarmUp(optimizer, config['lr_scheduler']['kwargs']['warmup_length'], T_max)
        else:
            scheduler = get_instance(torch.optim.lr_scheduler, "lr_scheduler", config, optimizer=optimizer)

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
        # For backward compatibility, some backbone initialization doesn't take device as argument
        try:
            backbone = get_instance(arch, "backbone", config, **{'device': self.device})
        except TypeError:
            backbone = get_instance(arch, "backbone", config)

        model = get_instance(arch, "classifier", config, **{'device': self.device, 'backbone': backbone}).to(self.device)

        if self.distribute:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device]
            )

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

        # Add DistributedSampler to each dataloader
        if self.distribute:
            for loaders in [train_loaders, test_loaders]:
                for i, dataloader in enumerate(loaders.dataloaders):
                    dataset = dataloader.dataset
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
                    loaders.dataloaders[i] = DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=dataloader.batch_size // self.config['n_gpu'],
                        num_workers=dataloader.num_workers,
                        drop_last=dataloader.drop_last
                    )

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
        method_name = self.config["classifier"]["name"]
        testing_times = self.config['testing_times']

        # 记录每个 task 的 average last accuracy
        batch_last_acc_list = np.zeros((self.task_num))
        task_last_acc_list = np.zeros((self.task_num))

        # 记录每个 task 的 best last accuracy
        best_batch_last_acc_list = np.zeros((self.task_num))
        best_task_last_acc_list = np.zeros((self.task_num))

        acc_table = np.zeros((self.task_num, self.task_num)) # A numpy array with shape [task_num, task_num], where [i, j] is acc of model on task j after learning task i
        bwt_list, frgt_list = [], []

        model = self.model.module if self.distribute else self.model
        
        if method_name == 'RAPF':
            model.model.classes_names = self.train_loader.cls_map

        for task_idx in range(self.task_num):
            self.task_idx = task_idx
            if self.rank == 0:
                print(f"================Task {task_idx} Start!================")
            
            if hasattr(model, 'before_task'):
                model.before_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))
            
            if self.rank == 0:
                print(f"Trainable Parameters for Task {task_idx} : {count_parameters(model)} / {count_all_parameters(model)} ({count_parameters(model)*100/count_all_parameters(model):.2f}%)")

            _, _, self.optimizer, self.scheduler = self._init_optim(self.config)
            dataloader = self.train_loader.get_loader(task_idx)

            if method_name == "bic":

                w_decay = 2e-4 * self.task_num / (task_idx + 1) # in source code?
                self.optimizer = optim.SGD(model.get_parameters(self.config), lr = 0.1, momentum = 0.9, weight_decay = w_decay)
                self.scheduler = MultiStepLR(self.optimizer, milestones = [100, 150, 200], gamma = 0.1)

                dataloader, val_bias_dataloader = self.model.spilt_and_update(dataloader, self.buffer, task_idx, self.config)

            elif isinstance(self.buffer, (LinearBuffer, LinearHerdingBuffer)) and self.buffer.buffer_size > 0 and task_idx > 0:
                datasets = dataloader.dataset
                if isinstance(datasets.images, list):
                    datasets.images.extend(self.buffer.images)
                    datasets.labels.extend(self.buffer.labels)
                elif isinstance(datasets.images, np.ndarray):
                    datasets.images = np.concatenate((datasets.images, self.buffer.images), axis=0)
                    datasets.labels = np.concatenate((datasets.labels, self.buffer.labels), axis=0)
                else:
                    assert 0

                dataloader = DataLoader(
                    datasets,
                    shuffle = True,
                    batch_size = self.config['batch_size'],
                    drop_last = False,
                    num_workers = self.config['num_workers']
                )
            
            if method_name in ["LoRAsub_DRS"]:
                print('Replacing Optim & Scheduler')
                self.optimizer = self.model.get_optimizer(self.config['optimizer']['kwargs']['lr'], self.config['optimizer']['kwargs']['weight_decay'])
                self.scheduler = CosineSchedule(self.optimizer, K=self.config['epoch'])

            if method_name == 'CL_LoRA':
                self.model.set_optim(self.optimizer)

            if self.rank == 0:
                print(f"================Task {task_idx} Training!================")
                print(f"The training samples number : {len(dataloader.dataset)}")
            
            # Reset Best Record
            best_batch_last_acc, best_task_last_acc = 0., 0.
            best_bwt, best_frgt = float('-inf'), float('inf')

            for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):
                if self.rank == 0:
                    print("================Train on train set================")
                train_meter = self._train(epoch_idx, dataloader)

                acc1 = torch.tensor(train_meter.avg("acc1"), device=self.device)
                loss = torch.tensor(train_meter.avg("loss"), device=self.device)
                if self.distribute:
                    # Aggregate accuracy across all processes
                    dist.barrier()
                    dist.all_reduce(acc1, op=dist.ReduceOp.SUM)  # Sum accuracy across processes
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    acc1 = acc1 / self.config['n_gpu']  # Normalize by world size
                    loss = loss / self.config['n_gpu']
                    dist.barrier()

                acc1 = acc1.item()
                loss = loss.item()
                
                if self.rank == 0:
                    print(f"Epoch [{epoch_idx}/{self.init_epoch if task_idx == 0 else self.inc_epoch}] Learning Rate {self.scheduler.get_last_lr()}\t|\tLoss: {loss:.4f} \tAverage Acc: {acc1:.2f} ")

                if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1) == self.inc_epoch:
                    if self.rank == 0:
                        print(f"================Validation on test set================")

                    # Disable validation for some method
                    if method_name in ['TRGP', 
                                       'RanPAC', 
                                       'MInfLoRA2', 
                                       'MInfLoRA3', 
                                       'PRAKA', 
                                       'TRGP_CLIP', 
                                       'LoRAsub_DRS',
                                       'CL_LoRA'
                        ]:
                        if self.rank == 0:
                            print(f" * Disabled validation for this method")
                    else:
                        test_acc = self._validate(task_idx)

                        batch_last_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                        best_batch_last_acc = max(batch_last_acc, best_batch_last_acc)

                        task_last_acc = np.mean(per_task_acc)
                        best_task_last_acc = max(task_last_acc, best_task_last_acc)

                        frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                        best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                        if self.rank == 0:
                            print(f" * [Batch] Last Average Acc: {batch_last_acc:.2f} (Best: {best_batch_last_acc:.2f})")
                            print(f" * [Task] Last Average Acc: {task_last_acc:.2f} (Best: {best_task_last_acc:.2f})")
                            print(f" * Forgetting: {frgt:.3f} (Best: {best_frgt:.3f})")
                            print(f" * Backward Transfer: {bwt:.2f} (Best: {best_bwt:.2f})")
                            print(f" * Per-Task Acc: {per_task_acc}")
            
                if self.config['lr_scheduler']['name'] == "PatienceSchedule":
                    self.scheduler.step(train_meter.avg('loss'))
                    if self.scheduler.get_last_lr() < self.config['lr_scheduler']['kwargs']['stopping_lr']:
                        if self.rank == 0:
                            print(f"{self.scheduler.get_last_lr()} < {self.config['lr_scheduler']['kwargs']['stopping_lr']}, stopping this task now")
                        break
                else:
                    self.scheduler.step()

            if hasattr(model, 'after_task'):
                model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

            # Update Buffer
            if method_name not in ['bic', 'ERACE', 'ERAML']:
                self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
                if self.buffer.buffer_size > 0:
                    if self.buffer.strategy == 'herding':
                        herding_update(self.train_loader.get_loader(task_idx).dataset, self.buffer, model.backbone, self.device)
                    elif self.buffer.strategy == 'random':
                        random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)
                    elif self.buffer.strategy == 'balance_random':
                        balance_random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)

            # Stage 2 Training : BIC (Stage 2 start after buffer being updated)
            if self.config["classifier"]["name"] == "bic" and task_idx > 0:

                bias_scheduler = optim.lr_scheduler.LambdaLR(model.bias_optimizer, lr_lambda=lambda e: 1)

                for epoch_idx in range(self.stage2_epoch):
                    if self.rank == 0:
                        print("================ Train on the train set (stage2)================")
                    train_meter = self.stage2_train(epoch_idx, val_bias_dataloader)

                    if self.rank == 0:
                        print(f"Epoch [{epoch_idx}/{self.stage2_epoch}] Learning Rate {bias_scheduler.get_last_lr()}\t|\tLoss: {train_meter.avg('loss'):.4f} \tAverage Acc: {train_meter.avg('acc1'):.2f} ")

                    if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1) == self.inc_epoch:
                        if self.rank == 0:
                            print("================ Test on the test set (stage2)================")

                        test_acc = self._validate(task_idx)

                        batch_last_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                        best_batch_last_acc = max(batch_last_acc, best_batch_last_acc)

                        task_last_acc = np.mean(per_task_acc)
                        best_task_last_acc = max(task_last_acc, best_task_last_acc)

                        frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                        best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                        if self.rank == 0:
                            print(f" * [Batch] Last Average Acc: {batch_last_acc:.2f} (Best: {best_batch_last_acc:.2f})")
                            print(f" * [Task] Last Average Acc: {task_last_acc:.2f} (Best: {best_task_last_acc:.2f})")
                            print(f" * Forgetting: {frgt:.3f} (Best: {best_frgt:.3f})")
                            print(f" * Backward Transfer: {bwt:.2f} (Best: {best_bwt:.2f})")
                            print(f" * Per-Task Acc: {per_task_acc}")

                    #bias_scheduler.step()

            for test_idx in range(testing_times):
                if self.rank == 0:
                    print(f"================Test {test_idx+1}/{testing_times} of Task {task_idx}!================")

                test_acc = self._validate(task_idx)

                batch_last_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                best_batch_last_acc = max(batch_last_acc, best_batch_last_acc)

                task_last_acc = np.mean(per_task_acc)
                best_task_last_acc = max(task_last_acc, best_task_last_acc)

                frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                if self.rank == 0:
                    print(f" * [Batch] Last Average Acc: {batch_last_acc:.2f} (Best: {best_batch_last_acc:.2f})")
                    print(f" * [Task] Last Average Acc: {task_last_acc:.2f} (Best: {best_task_last_acc:.2f})")
                    print(f" * Forgetting: {frgt:.3f} (Best: {best_frgt:.3f})")
                    print(f" * Backward Transfer: {bwt:.2f} (Best: {best_bwt:.2f})")
                    print(f" * Per-Task Acc: {per_task_acc}")

                batch_last_acc_list[task_idx] += batch_last_acc # avg_acc_list[task_idx] += avg_acc
                task_last_acc_list[task_idx] += task_last_acc
                acc_table[task_idx][:task_idx + 1] += np.array(per_task_acc)

            best_batch_last_acc_list[task_idx] = best_batch_last_acc
            best_task_last_acc_list[task_idx] = best_task_last_acc

            # Take mean of testing_times
            batch_last_acc_list[task_idx] /= testing_times
            task_last_acc_list[task_idx] /= testing_times
            acc_table[task_idx] /= testing_times

            batch_last_acc = batch_last_acc_list[task_idx]
            task_last_acc = task_last_acc_list[task_idx]

            frgt, bwt = compute_frgt(acc_table, acc_table[task_idx], task_idx), compute_bwt(acc_table, acc_table[task_idx], task_idx)
            best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)
            if task_idx > 1:
                frgt_list.append(frgt)
                bwt_list.append(bwt)
                
            if self.rank == 0:
                print(f"================Result of Task {task_idx} Testing!================")
                print(f" * [Batch] Last Average Acc: {batch_last_acc:.2f} (Best: {best_batch_last_acc:.2f})")
                print(f" * [Task] Last Average Acc: {task_last_acc:.2f} (Best: {best_task_last_acc:.2f})")
                print(f" * Forgetting: {frgt:.3f} (Best: {best_frgt:.3f})")
                print(f" * Backward Transfer: {bwt:.2f} (Best: {best_bwt:.2f})")
                print(f" * Per-Task Acc: {acc_table[task_idx][:task_idx + 1]}")

        batch_ovr_avg_acc = np.mean(batch_last_acc_list) #batch_ovr_avg_acc = np.mean(avg_acc_list)
        best_batch_ovr_avg_acc = np.mean(best_batch_last_acc_list) # best_batch_ovr_avg_acc = np.mean(best_avg_acc_list)
         
        task_ovr_avg_acc = np.sum(np.sum(acc_table[:task_idx + 1], axis = 1) / np.arange(1, task_idx + 2)) / (task_idx + 1)
        
        ovr_bwt = np.mean(bwt_list) if len(bwt_list) > 0 else float('-inf')
        ovr_frgt = np.mean(frgt_list) if len(frgt_list) > 0 else float('inf')

        if self.rank == 0:
            print(f"================Overall Result of {self.task_num} Tasks!================")
            print(f" * [Batch] Last Average Acc: {batch_last_acc:.2f} (Best: {best_batch_last_acc:.2f})")
            print(f" * [Task] Last Average Acc: {task_last_acc:.2f} (Best: {best_task_last_acc:.2f})")
            print(f" * Forgetting: {frgt:.3f} (Best: {best_frgt:.3f})")
            print(f" * Backward Transfer: {bwt:.2f} (Best: {best_bwt:.2f})")
            print(f" * [Batch] Overall Avg Acc : {batch_ovr_avg_acc:.2f} (Best: {best_batch_ovr_avg_acc:.2f})")
            print(f" * [Task] Overall Avg Acc : {task_ovr_avg_acc:.2f}")
            print(f" * Overall Frgt : {ovr_frgt:.3f}")
            print(f" * Overall BwT : {ovr_bwt:.2f}")
            print(f" * Average Acc Table : \n{acc_table}")

            print(f"================Model Performance Analysis================")
            print(f" * Time Costs : {(time() - experiment_begin):.2f} sec")
            fps = compute_fps(model, self.config)
            avg_fps, best_fps = fps['avg_fps'], fps['best_fps']
            print(f" * Average FPS (Best FPS) : {avg_fps:.0f} ({best_fps:.0f})")

    def stage2_train(self, epoch_idx, dataloader):
        """
        The stage 2 train stage of method : BIC

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        model = self.model.module if self.distribute else self.model

        model.eval()
        for layer in model.bias_layers:
            layer.train()

        meter = self.train_meter
        meter.reset()
        
        total = len(dataloader)
        for b, batch in tqdm(enumerate(dataloader), total=total, disable=(self.rank != 0)):

            output, acc, loss = model.stage2(batch)
            
            meter.update("acc1", 100 * acc)
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
        model = self.model.module if self.distribute else self.model

        model.train()
        if self.config['classifier']['name'] == 'bic':
            for layer in model.bias_layers:
                layer.eval()
        
        meter = deepcopy(self.train_meter)
        meter.reset()

        total = len(dataloader)
        init_seed(self.config['seed'] + epoch_idx, self.config['deterministic']) # Ensure Reproducibility
        for b, batch in tqdm(enumerate(dataloader), total=total, disable=(self.rank != 0)):
            
            batch['batch_id'] = b

            # These method's LR is updated every iterations, not epochs
            if self.config['classifier']['name'] in ['MOE_ADAPTER4CL', 'DMNSP', 'DMNSP_CIL']:
                self.scheduler.step(total * epoch_idx + b)

            if self.config["classifier"]["name"] in ['TRGP', 'DMNSP', 'DMNSP_CIL', 'TRGP_CLIP', 
                                                    'GPM', 'MoE_Test2', 'API', 'L2P']:
                self.optimizer.zero_grad()
                output, acc, loss = model.observe(batch)
            elif self.config["classifier"]["name"] in ['bic']:
                output, acc, loss = model.observe(batch)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
            else:
                output, acc, loss = model.observe(batch)
                self.optimizer.zero_grad()
                loss.backward()

            self.optimizer.step()

            if self.config["classifier"]["name"] in ['ERACE', 'ERAML']:
                model.add_reservoir()

            meter.update("acc1", 100 * acc)
            meter.update("loss", loss.item())

        return meter

    def _validate(self, task_idx):

        dataloaders = self.test_loader.get_loader(task_idx)

        model = self.model.module if self.distribute else self.model
        model.eval()

        if self.config["classifier"]["name"] == 'bic':
            for layer in model.bias_layers:
                layer.eval()

        per_task_acc = []
        count_all, correct_all = 0, 0

        if self.config['testing_per_task']:

            count_task, correct_task = 0, 0

            with torch.no_grad():
                for t, dataloader in enumerate(dataloaders):
                    correct_task, count_task = 0, 0

                    for b, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc = f"Testing on Task {t} data", disable=self.rank != 0):  # Disable tqdm for non-master processes
                        
                        if self.config['setting'] == 'task-aware':
                            output, acc = model.inference(batch, task_id=t)
                        elif self.config['setting'] == 'task-agnostic':
                            output, acc = model.inference(batch)
                        
                        correct_task += int(acc * batch['label'].shape[0])
                        count_task += batch['label'].shape[0]

                    correct_all += correct_task
                    count_all += count_task

                    if self.distribute:
                        pass

                    per_task_acc.append(round(correct_task * 100 / count_task, 2))

            if self.distribute:
                pass

        else:

            datasets = [dl.dataset for dl in dataloaders]

            all_images = np.concatenate([ds.images for ds in datasets], axis=0)
            all_labels = np.concatenate([ds.labels for ds in datasets], axis=0)

            merged_dataset = copy.deepcopy(datasets[0])
            merged_dataset.images = all_images
            merged_dataset.labels = all_labels

            merged_loader = DataLoader(
                    merged_dataset,
                    shuffle = True,
                    batch_size = self.config['batch_size'],
                    drop_last = False,
                    num_workers = self.config['num_workers'],
                    pin_memory=False
                )

            class_boundaries = []
            start_cls = 0
            for t in range(task_idx + 1):
                n_cls = self.init_cls_num if t == 0 else self.inc_cls_num
                class_boundaries.append((start_cls, start_cls + n_cls))
                start_cls += n_cls

            correct_by_task = np.zeros(task_idx + 1, dtype=int)
            count_by_task = np.zeros(task_idx + 1, dtype=int)

            # 4. 推理
            with torch.no_grad():
                for b, batch in tqdm(enumerate(merged_loader), total=len(merged_loader), desc=f"Testing merged tasks <= {task_idx}", disable=self.rank != 0):

                    if self.config['setting'] == 'task-aware':
                        print('Mostly methods dont support this, set testing_per_task to False')
                        raise NotImplementedError
                        output, acc = model.inference(batch, task_id=None)
                    elif self.config['setting'] == 'task-agnostic':
                        output, acc = model.inference(batch)
                    preds = output.cpu().numpy()

                    labels = batch['label'].cpu().numpy()
                    correct_all += np.sum(preds == labels)

                    count_all += len(labels)

                    # 统计每个 task 的正确率
                    for t, (start, end) in enumerate(class_boundaries):
                        mask = (labels >= start) & (labels < end)
                        if np.any(mask):
                            correct_by_task[t] += np.sum(preds[mask] == labels[mask])
                            count_by_task[t] += np.sum(mask)

            per_task_acc = [round(c * 100 / n, 2) if n > 0 else 0 for c, n in zip(correct_by_task, count_by_task)]

        avg_acc = round(correct_all * 100 / count_all, 2)

        return {
            "avg_acc": avg_acc,
            "per_task_acc": per_task_acc
        }
