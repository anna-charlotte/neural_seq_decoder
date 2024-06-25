import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from neural_decoder.dataset import ExtendedSpeechDataset
from neural_decoder.neural_decoder_trainer import (
    _padding_extended,
    _padding_phoneme,
    getDataLoader,
    getDatasetLoaders,
)
from text2brain.models.rnn import TextToBrainGRU
from text2brain.models.utils import load_text2brain_model


class PhonemeDataset:
    def __init__(
        self, data: list[Dict], transform=None, kernel_len: int = 32, stride: int = 4, phoneme_cls: int = None
    ):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.kernel_len = kernel_len
        self.stride = stride
        self.phoneme_cls = phoneme_cls
        if phoneme_cls is not None:
            assert phoneme_cls in range(len(PHONE_DEF_SIL))

        self.neural_windows = []
        self.phonemes = []
        self.days = []
        self.logits = []

        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                signal = data[day]["sentenceDat"][trial]
                for i in range(0, signal.size(0) - self.kernel_len + 1, self.stride):
                    logits = data[day]["logits"][trial][int(i / self.stride)]
                    phoneme = np.argmax(logits)
                    if self.phoneme_cls is None or self.phoneme_cls == phoneme:
                        self.phonemes.append(phoneme)
                        self.logits.append(logits)
                        window = signal[i : i + self.kernel_len]
                        self.neural_windows.append(window)
                        self.days.append(day)

        self.n_trials = len(self.phonemes)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_window = torch.tensor(self.neural_windows[idx], dtype=torch.float32)
        phoneme = torch.tensor(self.phonemes[idx], dtype=torch.int32)
        logits = torch.tensor(self.logits[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_window,
            phoneme,
            logits,
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


def main(args: dict) -> None:

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = args["device"]
    n_batches = args["n_batches"]
    batch_size = args["batch_size"]

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    train_loader = getDataLoader(
        data=train_data, batch_size=batch_size, shuffle=True, collate_fn=None, dataset_cls=PhonemeDataset
    )
    phonemes = []

    for i, batch in enumerate(train_loader):
        neural_window, phoneme, logits, dayIdx = batch
        neural_window, phoneme, logits, dayIdx = (
            neural_window.to(device),
            phoneme.to(device),
            logits.to(device),
            dayIdx.to(device),
        )

    test_file = args["test_set_path"]
    with open(test_file, "rb") as handle:
        test_data = pickle.load(handle)

    test_loader = getDataLoader(
        data=test_data, batch_size=batch_size, shuffle=False, collate_fn=None, dataset_cls=PhonemeDataset
    )


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cpu"
    args[
        "train_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    args[
        "test_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 32
    args["n_batches"] = 10000
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    # args["bidirectional"] = True
    main(args)
