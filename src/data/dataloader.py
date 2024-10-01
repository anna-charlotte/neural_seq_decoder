import torch
from torch.utils.data import DataLoader


class MergedDataLoader:
    def __init__(self, loader1: DataLoader, loader2: DataLoader, prop1: float = None):
        if prop1 is None:
            prop1 = len(loader1.dataset) / (len(loader1.dataset) + len(loader2.dataset))
        else:
            assert 0.0 <= prop1 <= 1.0
        print(f"prop1 = {prop1}")

        assert (
            loader1.batch_size == loader2.batch_size
        ), f"Batch sizes of the two given data loaders are not equal: {loader1.batch_size} != {loader2.batch_size}"
        self.batch_size = loader1.batch_size

        self.loader1 = loader1
        self.loader2 = loader2

        self.prop1 = prop1

        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)

    def __iter__(self):
        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)
        return self

    def __next__(self):
        if torch.rand(1).item() < self.prop1:
            try:
                return next(self.iter1)
            except StopIteration:
                self.iter1 = iter(self.loader1)
                return next(self.iter1)
        else:
            try:
                return next(self.iter2)
            except StopIteration:
                self.iter2 = iter(self.loader2)
                return next(self.iter2)

    def __len__(self):
        return len(self.loader1) + len(self.loader2)
