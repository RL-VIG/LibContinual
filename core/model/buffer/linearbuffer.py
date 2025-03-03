import numpy as np


class LinearBuffer:
    def __init__(self, buffer_size, strategy, batch_size):
        
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.batch_size = batch_size
        self.total_classes = 0
        self.images, self.labels = [], []

    def is_empty(self):
        return len(self.labels) == 0

class LinearSpiltBuffer:
    def __init__(self, buffer_size, strategy, batch_size, val_ratio):
        
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.batch_size = batch_size
        self.val_ratio = 0.1
        self.total_classes = 0
        self.train_images, self.train_labels = [], []
        self.val_images, self.val_labels = [], []

    def is_empty(self):
        return len(self.train_labels) == 0