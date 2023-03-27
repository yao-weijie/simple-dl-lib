import numpy as np
from .dataset import NdarrayDataset


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        assert isinstance(dataset, NdarrayDataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexs = np.arange(len(self.dataset))
        self.split_idx = np.arange(batch_size, len(dataset), batch_size)
        if self.shuffle:
            np.random.shuffle(self.indexs)  # inplace operation

    def __iter__(self):
        for idx in np.array_split(self.indexs, self.split_idx):
            yield self.dataset[self.indexs[idx]]
