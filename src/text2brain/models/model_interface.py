import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToBrainInterface(ABC):
    @abstractmethod
    def train_one_epoch(
        self,
        text: torch.Tensor,
        brain_signal: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, dayIdx):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, file_path: Path) -> None:
        raise NotImplementedError
