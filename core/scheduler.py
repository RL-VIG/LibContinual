from torch.optim import Optimizer
import math

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(epoch = last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        if self.K == 1:
            return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (2-1)))
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
    
    def get_last_lr(self):
        return self.get_lr()

class CosineAnnealingWarmUp(_LRScheduler):

    def __init__(self, optimizer, warmup_length, T_max = 0, last_epoch = -1):
        self.warmup_length = warmup_length
        self.T_max = T_max
        self.last_epoch = last_epoch

        super().__init__(optimizer, last_epoch)

    def cosine_lr(self, base_lr):

        return base_lr * 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.T_max))

    def warmup_lr(self, base_lr):

        return base_lr * (self.last_epoch + 1) / self.warmup_length

    def get_lr(self):
        if self.last_epoch < self.warmup_length:
            return [self.warmup_lr(base_lr) for base_lr in self.base_lrs]
        else:
            return [self.cosine_lr(base_lr) for base_lr in self.base_lrs]
    
    def get_last_lr(self):
        assert self.T_max > 0, 'CosineAnnealingWarmUp is called with T_max <= 0, Check your code'
        return self.get_lr()

class PatienceSchedule(_LRScheduler):

    def __init__(self, optimizer, patience, factor):
        self.factor = factor      # Factor to reduce the learning rate
        self.patience = patience   # Number of epochs with no improvement
        self.best_loss = float('inf')  # Best loss seen so far
        self.counter = 0            # Counter for patience

        super().__init__(optimizer, -1)

    def step(self, current_loss = None, **kwargs):
        # Some scheduler step function is called with parameter epoch
        # use kwargs to save it and don't do anything to it

        if current_loss is None:
            return 0
        
        # Check if the current loss improved
        if current_loss < self.best_loss:
            self.best_loss = current_loss  # Update the best loss
            self.counter = 0  # Reset counter since we have an improvement
        else:
            
            self.counter += 1  # Increment counter if no improvement
        
        # If patience is exhausted, reduce the learning rate
        if self.counter >= self.patience:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= self.factor  # Reduce learning rate by the factor
            print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.5f}")
            self.counter = 0  # Reset counter after reducing learning rate

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    
class _LRSchedulerIter(object):
    def __init__(self, optimizer, iter_step, n_epochs, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        last_iter = (last_epoch+1)*iter_step - 1
        self.total_iters = iter_step * n_epochs
        schedule = np.ones((self.total_iters, len(self.optimizer.param_groups)))
        schedule = [schedule[:, i][:, np.newaxis]*self.base_lrs[i] for i in range(len(self.base_lrs))]
        self.schedule = np.concatenate(schedule, axis=1)
        self.step(last_iter + 1)
        self.last_epoch = last_epoch
        self.last_iter = last_iter

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, iter):
        raise NotImplementedError

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(iter)):
            param_group['lr'] = lr

class CosineSchedulerIter(_LRSchedulerIter):
    def __init__(self, base_value, final_value, optimizer, iter_step, n_epochs, last_epoch=-1, warmup_epochs=0, start_warmup_value=0, freeze_iters=0):
        self.final_value = final_value
        super().__init__(optimizer, iter_step, n_epochs, last_epoch)
        warmup_iters = warmup_epochs * iter_step

        n_groups = len(self.optimizer.param_groups)
        assert len(final_value)==n_groups and len(base_value)==n_groups, f'Please provide {n_groups} final_value and base_value.'

        freeze_schedule = np.zeros((freeze_iters)) + 1e-6
        freeze_schedule = np.repeat(freeze_schedule[:, np.newaxis], n_groups, axis=1) # iters, n_groups

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        # warmup_schedule = np.repeat(warmup_schedule, n_groups, axis=1) # iters, n_groups

        # can base_value and final_value be list? -> each param_group of the optimizer
        iters = np.arange(self.total_iters - warmup_iters - freeze_iters)
        # schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = [final_value[i] + 0.5 * (base_value[i] - final_value[i]) * (1 + np.cos(np.pi * iters / len(iters))) for i in np.arange(n_groups)]
        schedule = np.array(schedule).transpose()
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))
        
        assert len(self.schedule) == self.total_iters

    def get_lr(self, iter):
        if iter >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[iter]
            
    def get_last_lr(self):
            """Returns the last computed learning rate by this scheduler."""
            return [group['lr'] for group in self.optimizer.param_groups]