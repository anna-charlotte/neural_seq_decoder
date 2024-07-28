import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class VAEBase(ABC, nn.Module):
    def __init__(self, latent_dim, input_shape: Tuple[int, int, int], device: str):
        super(VAEBase, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor):
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor):
        pass
