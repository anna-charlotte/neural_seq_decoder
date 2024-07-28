import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class VAEBase(ABC, nn.Module):
    def __init__(self, latent_dim, input_shape: Tuple[int, int, int], classes: list, device: str):
        super(VAEBase, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.classes = classes
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

    def save_state_dict(self, path: str) -> None:
        print(f"Store model state dict to: {path}")
        torch.save(self.state_dict(), path)
