import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class T2BGenInterface(ABC):

    @classmethod
    @abstractmethod
    def load_model(cls, model_path: Path):
        raise NotImplementedError

    @abstractmethod
    def create_synthetic_phoneme_dataset(self, n_samples, neural_window_shape: Tuple[int, int, int]):
        raise NotImplementedError


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
