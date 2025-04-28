import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from collections.abc import Iterable

class ERBuffer(nn.Module):
    def __init__(self, capacity):
        super().__init__()

        # create placeholders for each item
        self.buffers = []

        self.cap = capacity
        self.buffer_size = capacity
        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0

        # defaults
        self.add = self.add_reservoir
        self.sample = self.sample_random

    def __len__(self):
        return self.current_index


    def add_buffer(self, name, dtype, size):
        """ used to add extra containers (e.g. for logit storage) """

        tmp = torch.zeros(size=(self.cap,) + size, dtype=dtype).to(self.device)
        self.register_buffer(f'b{name}', tmp)
        self.buffers += [f'b{name}']

    def _init_buffers(self, batch):
        created = 0

        for name, tensor in batch.items():
            bname = f'b{name}'
            if bname not in self.buffers:
                
                if not type(tensor) == torch.Tensor:
                    tensor = torch.from_numpy(np.array([tensor]))

                self.add_buffer(name, tensor.dtype, tensor.shape[1:])
                created += 1

                print(f'created buffer {name}\t {tensor.dtype}, {tensor.shape[1:]}')

        assert created in [0, len(batch)], 'not all buffers created at the same time'

    def add_reservoir(self, batch):

        self._init_buffers(batch)

        n_elem = batch['x'].shape[0]

        place_left = max(0, self.cap - self.current_index)

        indices = torch.FloatTensor(n_elem).to(self.device)
        indices = indices.uniform_(0, self.n_seen_so_far).long()

        if place_left > 0:
            upper_bound = min(place_left, n_elem)
            indices[:upper_bound] = torch.arange(upper_bound) + self.current_index

        valid_indices = (indices < self.cap).long()
        idx_new_data  = valid_indices.nonzero().squeeze(-1)
        idx_buffer    = indices[idx_new_data]

        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.cap)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def add_balanced(self, batch):
        self._init_buffers(batch)

        n_elem = batch['x'].size(0)

        # increment first
        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.cap)

        # first thing is we just add all the data
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')

            if not isinstance(data, Iterable):
                data = buffer.new(size=(n_elem, *buffer.shape[1:])).fill_(data)

            buffer = torch.cat((data, buffer))[:self.n_seen_so_far]
            setattr(self, f'b{name}', buffer)

        n_samples_over = buffer.size(0) - self.cap

        # no samples to remove
        if n_samples_over <= 0:
            return

        # remove samples from the most common classes
        class_count   = self.by.bincount()
        rem_per_class = torch.zeros_like(class_count)

        while rem_per_class.sum() < n_samples_over:
            max_idx = class_count.argmax()
            rem_per_class[max_idx] += 1
            class_count[max_idx]   -= 1

        # always remove the oldest samples for each class
        classes_trimmed = rem_per_class.nonzero().flatten()
        idx_remove = []

        for cls in classes_trimmed:
            cls_idx = (self.by == cls).nonzero().view(-1)
            idx_remove += [cls_idx[-rem_per_class[cls]:]]

        idx_remove = torch.cat(idx_remove)
        idx_mask   = torch.BoolTensor(buffer.size(0)).to(self.device)
        idx_mask.fill_(0)
        idx_mask[idx_remove] = 1

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')
            buffer = buffer[~idx_mask]
            setattr(self, f'b{name}', buffer)

    def add_queue(self, batch):
        self._init_buffers(batch)

        if not hasattr(self, 'queue_ptr'):
            self.queue_ptr = 0

        start_idx = self.queue_ptr
        end_idx   = (start_idx + batch['x'].size(0)) % self.cap

        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')
            buffer[start_idx:end_idx] = data

    def sample_random(self, amt, exclude_task=None, **kwargs):
        buffers = OrderedDict()

        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = torch.where(self.bt != exclude_task)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[:self.current_index]

        n_selected = buffers['x'].size(0)
        if n_selected <= amt:
            assert n_selected > 0
            return buffers
        else:
            idx_np = np.random.choice(buffers['x'].size(0), amt, replace=False)
            indices = torch.from_numpy(idx_np).to(self.bx.device)

            return OrderedDict({k:v[indices] for (k,v) in buffers.items()})

    def sample_balanced(self, amt, exclude_task=None, **kwargs):
        buffers = OrderedDict()

        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[:self.current_index]

        class_count = buffers['y'].bincount()

        # a sample's prob. of being sample is inv. prop to its class abundance
        class_sample_p = 1. / class_count.float() / class_count.size(0)
        per_sample_p   = class_sample_p.gather(0, buffers['y'])
        indices        = torch.multinomial(per_sample_p, amt)

        return OrderedDict({k:v[indices] for (k,v) in buffers.items()})

    def sample_pos_neg(self, inc_data, task_free=True, same_task_neg=True):

        x     = inc_data['x']                              
        label = inc_data['y']                               
        task  = torch.zeros_like(label).fill_(inc_data['t'])

        # we need to create an "augmented" buffer containing the incoming data
        bx   = torch.cat((self.bx[:self.current_index], x))    
        by   = torch.cat((self.by[:self.current_index], label)) 
        bt   = torch.cat((self.bt[:self.current_index], task)) 
        bidx = torch.arange(bx.size(0)).to(bx.device)     

        # buf_size x label_size
        same_label = label.view(1, -1)             == by.view(-1, 1)  
        same_task  = task.view(1, -1)              == bt.view(-1, 1)   
        same_ex    = bidx[-x.size(0):].view(1, -1) == bidx.view(-1, 1) 

        task_labels = label.unique()
        real_same_task = same_task

        if task_free:
            same_task = torch.zeros_like(same_task)

            for label_ in task_labels:
                label_exp = label_.view(1, -1).expand_as(same_task)
                same_task = same_task | (label_exp == by.view(-1, 1))

        valid_pos  = same_label & ~same_ex

        if same_task_neg:
            valid_neg = ~same_label & same_task
        else:
            valid_neg = ~same_label

        # remove points which don't have pos, neg from same and diff t
        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0

        invalid_idx = ~has_valid_pos | ~has_valid_neg

        if invalid_idx.sum() > 0:
            # so the fetching operation won't fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        # easier if invalid_idx is a binary tensor
        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        n_fwd = torch.stack((bidx[-x.size(0):], pos_idx, neg_idx), 1)[~invalid_idx].unique().size(0)

        return bx[pos_idx], \
               bx[neg_idx], \
               by[pos_idx], \
               by[neg_idx], \
               is_invalid,  \
               n_fwd

    def sample_minimal_pos_neg(self, inc_data, task_free=True, same_task_neg=True):
        """ maximize choosing the incoming data to minimize forward passes """

        x     = inc_data['x']
        label = inc_data['y']
        task  = torch.zeros_like(label).fill_(inc_data['t'])

        '''
        # we need to create an "augmented" buffer containing the incoming data
        bx   = torch.cat((self.bx[:self.current_index], x))
        by   = torch.cat((self.by[:self.current_index], label))
        bt   = torch.cat((self.bt[:self.current_index], task))
        bidx = torch.arange(bx.size(0)).to(bx.device)

        # buf_size x label_size
        same_label = label.view(1, -1)             == by.view(-1, 1)
        same_task  = task.view(1, -1)              == bt.view(-1, 1)
        same_ex    = bidx[-x.size(0):].view(1, -1) == bidx.view(-1, 1)
        '''

        bidx = torch.arange(x.size(0)).to(x.device)

        # label_size x label_size
        same_label = label.view(1, -1)             == label.view(-1, 1)
        same_task  = task.view(1, -1)              == task.view(-1, 1)
        same_ex    = bidx.view(1, -1)              == bidx.view(-1, 1)

        task_labels = label.unique()
        real_same_task = same_task

        # TASK FREE METHOD : instead of using the task ID, we'll use labels in
        # the current batch to mimic task
        if task_free:
            same_task = torch.zeros_like(same_task)

            for label_ in task_labels:
                label_exp = label_.view(1, -1).expand_as(same_task)
                same_task = same_task | (label_exp == label.view(-1, 1))

        valid_pos  = same_label & ~same_ex

        if same_task_neg:
            valid_neg = ~same_label & same_task
        else:
            valid_neg = ~same_label

        # remove points which don't have pos, neg from same and diff t
        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0

        invalid_idx = ~has_valid_pos | ~has_valid_neg

        if invalid_idx.any():
            # so the fetching operation won't fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        # easier if invalid_idx is a binary tensor
        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        # return
        pos_x, neg_x = x[pos_idx], x[neg_idx]
        pos_y, neg_y = label[pos_idx], label[neg_idx]

        n_fwd = torch.stack((bidx, pos_idx, neg_idx), 1)[~invalid_idx].unique().size(0)

        # --- handle cases that can be solved by looking into the buffer:
        if invalid_idx.any():
            # build new input
            invalid_data = OrderedDict()
            invalid_data['x'] = x[invalid_idx]
            invalid_data['y'] = label[invalid_idx]
            invalid_data['t'] = inc_data['t']

            n_pos_x, n_neg_x, n_pos_y, n_neg_y, n_is_invalid, n_new_fwd = \
                    self.sample_pos_neg(invalid_data, task_free=task_free, same_task_neg=same_task_neg)

            # next we fill the invalid indices with their potentially valid points from the buffer
            pos_x[invalid_idx][~n_is_invalid].data.copy_(n_pos_x[~n_is_invalid])
            neg_x[invalid_idx][~n_is_invalid].data.copy_(n_neg_x[~n_is_invalid])
            pos_y[invalid_idx][~n_is_invalid].data.copy_(n_pos_y[~n_is_invalid])
            neg_y[invalid_idx][~n_is_invalid].data.copy_(n_neg_y[~n_is_invalid])

            invalid_idx[invalid_idx].data.copy_(n_is_invalid)

            n_fwd += n_new_fwd

        return pos_x, neg_x, pos_y, neg_y, is_invalid, n_fwd
