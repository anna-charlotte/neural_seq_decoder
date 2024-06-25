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

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import getDataLoader



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
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 32
    args["n_batches"] = 10000
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    # args["bidirectional"] = True
    main(args)
