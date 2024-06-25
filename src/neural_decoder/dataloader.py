import torch


class MergedDataLoader:
    def __init__(self, loader1, loader2, prop1=0.5):
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
        return min(len(self.loader1), len(self.loader2))
