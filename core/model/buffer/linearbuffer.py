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

    

