from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from neural_decoder.phoneme_utils import PHONE_DEF_SIL


def _padding(batch: tuple) -> tuple:

    X, y, X_lens, y_lens, days = zip(*batch)

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


class BaseDataset(Dataset, ABC):
    def __init__(self, data: list[Dict], transform: callable = None) -> None:
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = None
        self.prepare_data()

    @abstractmethod
    def prepare_data(self):
        """
        Prepare any dataset-specific properties and structures.
        This method must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the total number of items in the dataset.
        This must be implemented by all subclasses and should use the `self.length` attribute.
        """
        return self.length

    @abstractmethod
    def __getitem__(self, idx):
        """
        Retrieve an item by its index.
        This method must be implemented by all subclasses.
        """
        pass


class SpeechDataset(BaseDataset):
    def prepare_data(self) -> None:
        self.n_trials = sum([len(d["sentenceDat"]) for d in self.data])
        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(self.data[day]["sentenceDat"])):
                self.neural_feats.append(self.data[day]["sentenceDat"][trial])
                self.phone_seqs.append(self.data[day]["phonemes"][trial])
                self.neural_time_bins.append(self.data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(self.data[day]["phoneLens"][trial])
                self.days.append(day)

    def __getitem__(self, idx) -> tuple:
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)
        if self.transform:
            neural_feats = self.transform(neural_feats)
        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )

    def __len__(self) -> int:
        return self.n_trials


class PhonemeDataset(BaseDataset):
    def __init__(
        self,
        data: list[Dict],
        transform: callable = None,
        kernel_len: int = 32,
        stride: int = 4,
        phoneme_cls: int = None,
    ) -> None:
        self.kernel_len = kernel_len
        self.stride = stride
        self.phoneme_cls = phoneme_cls
        super().__init__(data, transform)

    def prepare_data(self) -> None:
        if self.phoneme_cls is not None:
            assert self.phoneme_cls in range(len(PHONE_DEF_SIL))

        self.neural_windows = []
        self.phonemes = []
        self.days = []
        self.logits = []

        for day in range(self.n_days):
            for trial in range(len(self.data[day]["sentenceDat"])):
                signal = self.data[day]["sentenceDat"][trial]
                for i in range(0, signal.size(0) - self.kernel_len + 1, self.stride):
                    logits = self.data[day]["logits"][trial][int(i / self.stride)]
                    phoneme = np.argmax(logits)
                    if self.phoneme_cls is None or self.phoneme_cls == phoneme:
                        self.phonemes.append(phoneme)
                        self.logits.append(logits)
                        window = signal[i : i + self.kernel_len]
                        self.neural_windows.append(window)
                        self.days.append(day)

        self.n_trials = len(self.phonemes)

    def __getitem__(self, idx) -> tuple:
        neural_window = torch.tensor(self.neural_windows[idx], dtype=torch.float32)
        phoneme = torch.tensor(self.phonemes[idx], dtype=torch.int32)
        logits = torch.tensor(self.logits[idx], dtype=torch.float32)

        if self.transform:
            neural_window = self.transform(neural_window)

        return (
            neural_window,
            phoneme,
            logits,
            torch.tensor(self.days[idx], dtype=torch.int64),
        )

    def __len__(self) -> int:
        return self.n_trials
