import os
import sys
import torch
import numpy as np
import core.model as arch
import torch.optim as optim

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
from pprint import pprint
from core.scheduler import CosineSchedule, PatienceSchedule, CosineAnnealingWarmUp

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
        log_prefix = log_prefix.replace("/", "-")
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

        if config['device_ids'] == 'auto':

            least_utilized_device = 0
            most_free_mem_perc = float('-inf')

            for device_id in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                if total_mem > 0 and free_mem/total_mem > most_free_mem_perc:
                    least_utilized_device = device_id
                    most_free_mem_perc = free_mem/total_mem

            print(f'Using GPU {least_utilized_device}')
            device = torch.device(f'cuda:{least_utilized_device}')
            torch.cuda.empty_cache()

        else:
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

        optimizer = get_instance(
            torch.optim, "optimizer", config, params=self.model.get_parameters(config)
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
        # TODO: For backward compatibility, some backbone initialization doesn't take device as argument
        try:
            backbone = get_instance(arch, "backbone", config, **{'device': self.device})
        except TypeError:
            backbone = get_instance(arch, "backbone", config)

        model = get_instance(arch, "classifier", config, **{'device': self.device, 'backbone': backbone}).to(self.device)

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
        method_name = self.config["classifier"]["name"]
        testing_times = self.config['testing_times']

        avg_acc_list = np.zeros((self.task_num))
        best_avg_acc_list = np.zeros((self.task_num))
        acc_table = np.zeros((self.task_num, self.task_num)) # A numpy array with shape [task_num, task_num], where [i, j] is acc of model on task j after learning task i
        bwt_list, frgt_list = [], []
        
        if method_name == 'RAPF':
            self.model.model.classes_names = self.train_loader.cls_map

        for task_idx in range(self.task_num):
            self.task_idx = task_idx
            print(f"================Task {task_idx} Start!================")
            
            if hasattr(self.model, 'before_task'):
                self.model.before_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))
            
            print(f"Trainable Parameters for Task {task_idx} : {count_parameters(self.model)} / {count_all_parameters(self.model)} ({count_parameters(self.model)*100/count_all_parameters(self.model):.2f}%)")

            _, _, self.optimizer, self.scheduler = self._init_optim(self.config)
            dataloader = self.train_loader.get_loader(task_idx)

            if method_name == "bic":

                w_decay = 2e-4 * self.task_num / (task_idx + 1) # in source code?
                self.optimizer = optim.SGD(self.model.get_parameters(self.config), lr = 0.1, momentum = 0.9, weight_decay = w_decay)
                self.scheduler = MultiStepLR(self.optimizer, milestones = [100, 150, 200], gamma = 0.1)

                dataloader, val_bias_dataloader = bic.spilt_and_update11(dataloader, self.buffer, task_idx, self.config)

            elif isinstance(self.buffer, (LinearBuffer, LinearHerdingBuffer)) and self.buffer.buffer_size > 0 and task_idx > 0:
                datasets = dataloader.dataset
                datasets.images.extend(self.buffer.images)
                datasets.labels.extend(self.buffer.labels)
                dataloader = DataLoader(
                    datasets,
                    shuffle = True,
                    batch_size = self.config['batch_size'],
                    drop_last = False,
                    num_workers = self.config['num_workers']
                )

            print(f"================Task {task_idx} Training!================")
            print(f"The training samples number : {len(dataloader.dataset)}")

            best_avg_acc, best_bwt, best_frgt = 0., float('-inf'), float('inf')
            for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):
                
                print("================Train on train set================")
                train_meter = self._train(epoch_idx, dataloader)
                print(f"Epoch [{epoch_idx}/{self.init_epoch if task_idx == 0 else self.inc_epoch}] Learning Rate {self.scheduler.get_last_lr()}\t|\tLoss: {train_meter.avg('loss'):.4f} \tAverage Acc: {train_meter.avg('acc1'):.2f} ")

                if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1) == self.inc_epoch:
                    print(f"================Validation on test set================")

                    # Disable validation for some method
                    if method_name in ['TRGP', 'RanPAC', 'MInfLoRA', 'MInfLoRA2', 'MInfLoRA3', 'PRAKA', 'TRGP_CLIP'] or 'MInfLoRA' in method_name:
                        print(f" * Disabled validation for this method")
                    else:
                        test_acc = self._validate(task_idx)

                        avg_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                        best_avg_acc = max(avg_acc, best_avg_acc)

                        frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                        best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                        print(f" * [Batch] Last Average Acc (Best Last Average Acc) : {avg_acc:.2f} ({best_avg_acc:.2f})")
                        print(f" * Forgetting (Best Forgetting) : {frgt:.3f} ({best_frgt:.3f})")
                        print(f" * Backward Transfer (Best Backward Transfer) : {bwt:.2f} ({best_bwt:.2f})")
                        print(f" * Per-Task Acc : {per_task_acc}")
            
                if self.config['lr_scheduler']['name'] == "PatienceSchedule":
                    self.scheduler.step(train_meter.avg('loss'))
                    if self.scheduler.get_last_lr() < self.config['lr_scheduler']['kwargs']['stopping_lr']:
                        print(f"{self.scheduler.get_last_lr()} < {self.config['lr_scheduler']['kwargs']['stopping_lr']}, stopping this task now")
                        break
                else:
                    self.scheduler.step()

                if self.config['classifier']['name'] == 'MInfLoRA3':
                    self.model.mid_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

            if hasattr(self.model, 'after_task'):
                self.model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

            # Update Buffer
            if method_name != 'bic':
                self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num
                if self.buffer.buffer_size > 0:
                    if self.buffer.strategy == 'herding':
                        herding_update(self.train_loader.get_loader(task_idx).dataset, self.buffer, self.model.backbone, self.device)
                    elif self.buffer.strategy == 'random':
                        random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)
                    elif self.buffer.strategy == 'balance_random':
                        balance_random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)

            # Stage 2 Training : BIC (Stage 2 start after buffer being updated)
            if self.config["classifier"]["name"] == "bic" and task_idx > 0:

                bias_scheduler = optim.lr_scheduler.LambdaLR(self.model.bias_optimizer, lr_lambda=lambda e: 1)

                for epoch_idx in range(self.stage2_epoch):
                    print("================ Train on the train set (stage2)================")
                    train_meter = self.stage2_train(epoch_idx, val_bias_dataloader)

                    print(f"Epoch [{epoch_idx}/{self.stage2_epoch}] Learning Rate {bias_scheduler.get_last_lr()}\t|\tLoss: {train_meter.avg('loss'):.4f} \tAverage Acc: {train_meter.avg('acc1'):.2f} ")

                    if (epoch_idx+1) % self.val_per_epoch == 0 or (epoch_idx+1) == self.inc_epoch:
                        print("================ Test on the test set (stage2)================")

                        test_acc = self._validate(task_idx)

                        avg_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                        best_avg_acc = max(avg_acc, best_avg_acc)

                        frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                        best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                        print(f" * [Batch] Last Average Acc (Best Last Average Acc) : {avg_acc:.2f} ({best_avg_acc:.2f})")
                        print(f" * Forgetting (Best Forgetting) : {frgt:.3f} ({best_frgt:.3f})")
                        print(f" * Backward Transfer (Best Backward Transfer) : {bwt:.2f} ({best_bwt:.2f})")
                        print(f" * Per-Task Acc : {per_task_acc}")
            
                    #bias_scheduler.step()

            for test_idx in range(testing_times):
                print(f"================Test {test_idx+1}/{testing_times} of Task {task_idx}!================")

                test_acc = self._validate(task_idx)
                avg_acc, per_task_acc = test_acc['avg_acc'], test_acc['per_task_acc']
                best_avg_acc = max(avg_acc, best_avg_acc)

                frgt, bwt = compute_frgt(acc_table, per_task_acc, task_idx), compute_bwt(acc_table, per_task_acc, task_idx)
                best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)

                print(f" * [Batch] Last Average Acc (Best Last Average Acc) : {avg_acc:.2f} ({best_avg_acc:.2f})")
                print(f" * Forgetting (Best Forgetting) : {frgt:.3f} ({best_frgt:.3f})")
                print(f" * Backward Transfer (Best Backward Transfer) : {bwt:.2f} ({best_bwt:.2f})")
                print(f" * Per-Task Acc : {per_task_acc}")

                avg_acc_list[task_idx] += avg_acc
                acc_table[task_idx][:task_idx + 1] += np.array(per_task_acc)

            # Take mean of testing_times
            best_avg_acc_list[task_idx] = best_avg_acc
            avg_acc_list[task_idx] /= testing_times
            acc_table[task_idx] /= testing_times 

            #avg_acc = np.mean(acc_table[task_idx][:task_idx + 1])
            avg_acc = avg_acc_list[task_idx]

            frgt, bwt = compute_frgt(acc_table, acc_table[task_idx], task_idx), compute_bwt(acc_table, acc_table[task_idx], task_idx)
            best_frgt, best_bwt = min(frgt, best_frgt), max(bwt, best_bwt)
            if task_idx > 1:
                frgt_list.append(frgt)
                bwt_list.append(bwt)
                
            print(f"================Result of Task {task_idx} Testing!================")
            print(f" * [Batch] Last Average Acc (Best Last Average Acc) : {avg_acc:.2f} ({best_avg_acc:.2f})")
            print(f" * Forgetting (Best Forgetting) : {frgt:.3f} ({best_frgt:.3f})")
            print(f" * Backward Transfer (Best Backward Transfer) : {bwt:.2f} ({best_bwt:.2f})")
            print(f" * Per-Task Acc : {acc_table[task_idx][:task_idx + 1]}")

        batch_ovr_avg_acc = np.mean(avg_acc_list)
        best_batch_ovr_avg_acc = np.mean(best_avg_acc_list)
        task_ovr_avg_acc = np.sum(np.sum(acc_table[:task_idx + 1], axis = 1) / np.arange(1, task_idx + 2)) / (task_idx + 1)
        
        ovr_bwt = np.mean(bwt_list) if len(bwt_list) > 0 else float('-inf')
        ovr_frgt = np.mean(frgt_list) if len(frgt_list) > 0 else float('inf')

        print(f"================Overall Result of {self.task_num} Tasks!================")
        print(f" * [Batch] Last Average Acc (Best Last Average Acc) : {avg_acc:.2f} ({best_avg_acc:.2f})")
        print(f" * Forgetting (Best Forgetting) : {frgt:.3f} ({best_frgt:.3f})")
        print(f" * Backward Transfer (Best Backward Transfer) : {bwt:.2f} ({best_bwt:.2f})")
        print(f" * [Batch] Overall Avg Acc : {batch_ovr_avg_acc:.2f} ({best_batch_ovr_avg_acc:.2f})")
        print(f" * [Task] Overall Avg Acc : {task_ovr_avg_acc:.2f}")
        print(f" * Overall Frgt : {ovr_frgt:.3f}")
        print(f" * Overall BwT : {ovr_bwt:.2f}")
        print(f" * Average Acc Table : \n{acc_table}")

        print(f"================Model Performance Analysis================")
        print(f" * Time Costs : {(time() - experiment_begin):.2f} sec")
        fps = compute_fps(self.model, self.config)
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

        self.model.eval()
        for layer in self.model.bias_layers:
            layer.train()

        meter = self.train_meter
        meter.reset()
        
        total = len(dataloader)
        for b, batch in tqdm(enumerate(dataloader), total=total):

            output, acc, loss = self.model.stage2(batch)
            
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
        self.model.train()
        if self.config['classifier']['name'] == 'bic':
            for layer in self.model.bias_layers:
                layer.eval()
        
        meter = deepcopy(self.train_meter)
        meter.reset()

        total = len(dataloader)
        for b, batch in tqdm(enumerate(dataloader), total=total):

            batch['batch_id'] = b
            # These method's LR is updated every iterations, not epochs
            if self.config['classifier']['name'] in ['MOE_ADAPTER4CL', 'DMNSP', 'DMNSP_CIL']:
                self.scheduler.step(total * epoch_idx + b)

            if self.config["classifier"]["name"] in ['TRGP', 'DMNSP', 'DMNSP_CIL', 'TRGP_CLIP', 
                                                    'GPM', 'MoE_Test2', 'API', 'L2P']:
                self.optimizer.zero_grad()
                output, acc, loss = self.model.observe(batch)
            elif self.config["classifier"]["name"] in ['bic']:
                output, acc, loss = self.model.observe(batch)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
            else:
                output, acc, loss = self.model.observe(batch)
                self.optimizer.zero_grad()
                loss.backward()

            self.optimizer.step()

            meter.update("acc1", 100 * acc)
            meter.update("loss", loss.item())

        return meter

    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        if self.config["classifier"]["name"] == 'bic':
            for layer in self.model.bias_layers:
                layer.eval()

        total_meter = deepcopy(self.test_meter)
        meter = deepcopy(self.test_meter)
        
        total_meter.reset()
        meter.reset()
        
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter.reset()

                for batch in tqdm(dataloader, desc = f"Testing on Task {t} data"):
                    
                    if self.config['setting'] == 'task-aware':
                        output, acc = self.model.inference(batch, task_id=t)
                    elif self.config['setting'] == 'task-agnostic':
                        output, acc = self.model.inference(batch)
                    
                    meter.update("acc1", 100 * acc)
                    total_meter.update("acc1", 100 * acc)
                
                per_task_acc.append(round(meter.avg("acc1"), 2))

        return {"avg_acc" : round(total_meter.avg("acc1"), 2), 
                "per_task_acc" : per_task_acc}
