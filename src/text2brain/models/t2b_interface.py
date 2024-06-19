import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class TextToBrainInterface(ABC):

    @abstractmethod
    def train_one_epoch(
        self,
        text: torch.Tensor,
        brain_signal: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, text):
        raise NotImplementedError
        
    @abstractmethod
    def save_weights(self, save_to_dir: Path):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, file_dir: Path) -> None:
        raise NotImplementedError