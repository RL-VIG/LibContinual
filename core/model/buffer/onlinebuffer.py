import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from https://github.com/gydpku/OCM/blob/main/buffer.py

class OnlineBuffer(nn.Module):
    def __init__(self, buffer_size, batch_size, input_size):
        super().__init__()

        self.place_left = True
        self.strategy = None
        self.buffer_size = buffer_size
        print('buffer has %d slots' % buffer_size, buffer_size)

        buf_data = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        buf_targets = torch.LongTensor(buffer_size).fill_(0)
        buf_tasks = torch.LongTensor(buffer_size).fill_(0)

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        self.total_classes = 0
        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buf_data', buf_data)
        self.register_buffer('buf_targets', buf_targets)
        self.register_buffer('buf_tasks', buf_tasks)


    def tensor_to_device(self, device):
        self.device = device
        self.buf_data.to(device)
        self.buf_targets.to(device)
        self.buf_tasks.to(device)



    def add_reservoir(self, x, y, task):
        n_elem = x.size(0)
        
        self.device = x.device
        place_left = max(0, self.buffer_size - self.current_index)
        offset = min(place_left, n_elem)

        if place_left:
            offset = min(place_left, n_elem)

            self.buf_data[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.buf_targets[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.buf_tasks[self.current_index: self.current_index + offset].fill_(task)
            self.current_index += offset
            self.n_seen_so_far += offset

            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.buf_data.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.buf_data.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.buf_targets.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.buf_tasks.size(0), pdb.set_trace()

        assert idx_new_data.max() < x.size(0), pdb.set_trace()
        assert idx_new_data.max() < y.size(0), pdb.set_trace()

        if self.buf_data.device != x.device:
            self.buf_data = self.buf_data.to(x.device)
            self.buf_targets = self.buf_targets.to(x.device)
            self.buf_tasks = self.buf_tasks.to(x.device)

        self.buf_data[idx_buffer] = x[idx_new_data]
        self.buf_targets[idx_buffer] = y[idx_new_data]
        self.buf_tasks[idx_buffer] = task




    def sample(self, amount, exclude_task = None, ret_ind = False):

        if self.buf_data.device != self.device:
            self.buf_data = self.buf_data.to(self.device)
            self.buf_targets = self.buf_targets.to(self.device)
            self.buf_tasks = self.buf_tasks.to(self.device)

        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bt = self.buf_data[valid_indices], self.buf_targets[valid_indices], self.buf_tasks[valid_indices]
        else:
            bx, by, bt = self.buf_data[:self.current_index], self.buf_targets[:self.current_index], self.buf_tasks[:self.current_index]

        if bx.size(0) < amount:
            if ret_ind:
                return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amount, replace=False))
            indices = indices.to(self.device)

            if ret_ind:
                return bx[indices], by[indices], bt[indices], indices
            else:
                return bx[indices], by[indices], bt[indices]