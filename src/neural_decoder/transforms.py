import torch.nn as nn


class SoftsignTransform:
    def __init__(self):
        self.softsign = nn.Softsign()

    def __call__(self, sample):
        return self.softsign(sample)
