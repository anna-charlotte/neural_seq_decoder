import torch.nn as nn

from neural_decoder.phoneme_utils import reorder_neural_window


class SoftsignTransform:
    """
    The softsign transform maps the input to a range between -1 and 1.
    """

    def __init__(self):
        self.softsign = nn.Softsign()

    def __call__(self, sample):
        return self.softsign(sample)


class ReorderChannelTransform:
    """
    Reorder the channels from area 6v by the given ordering.
    """

    def __init__(self):
        self.reorder = reorder_neural_window

    def __call__(self, sample):
        return self.reorder(sample)


class TransposeTransform:
    def __call__(self, x):
        return x.transpose(0, 1)


class AddOneDimensionTransform:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        return x.unsqueeze(self.dim)
