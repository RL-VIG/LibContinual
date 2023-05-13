import numpy as np


class LinearBuffer:
    def __init__(self, buffer_size, strategy, batch_size):
        
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.batch_size = batch_size
        self.images, self.labels = [], []

    def update(self, datasets, task_idx):
        if self.buffer_size <= 0:
            print("No Buffer need update!")
            return
        
        if self.strategy == 'random':
            self.random_update(datasets, task_idx)
        else:
            raise Exception("Un implemented type in buffer.strategy")
        

    def random_update(self, datasets, task_idx):
        images = np.array(self.images + datasets.images)
        labels = np.array(self.labels + datasets.labels)
        perm = np.random.permutation(len(labels))

        images, labels = images[perm[:self.buffer_size]], labels[perm[:self.buffer_size]]

        self.images, self.labels = images.tolist(), labels.tolist()

    def is_empty(self):
        return len(self.labels) == 0
        
    

