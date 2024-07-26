from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
import pickle
from pathlib import Path

class VAEBase(ABC, nn.Module):
    def __init__(self, latent_dim, input_shape: Tuple[int, int, int]):
        super(VAEBase, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor):
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor):
        pass
