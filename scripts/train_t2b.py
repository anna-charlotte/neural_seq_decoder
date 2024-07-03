import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from neural_decoder.neural_decoder_trainer import get_dataset_loaders
from text2brain.models.rnn import TextToBrainGRU
from text2brain.models.utils import load_text2brain_model


def main(args: dict) -> None:

    os.makedirs(args["output_dir"], exist_ok=True)

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = args["device"]
    n_batches = args["n_batches"]
    batch_size = args["batch_size"]

    print("Get data loaders ...")
    train_loader, test_loader, loaded_data = get_dataset_loaders(args["dataset_path"], batch_size,)

    # Parameters
    input_dim = args["n_input_features"]
    hidden_dim = args["hidden_dim"]
    output_dim = args["n_output_features"]
    n_layers = args["n_layers"]

    # Initialize the model, loss function, and optimizer
    model = TextToBrainGRU(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers,
    )

    args["model_class"] = model.__class__.__name__

    model_dir = Path(args["output_dir"]) / "model"
    args_file = model_dir / "args"
    with open(args_file, "wb") as file:
        print(f"Write args to: {args_file}")
        pickle.dump(args, file)

    model = load_text2brain_model(model_dir=model_dir)
    min_loss = float("inf")

    for i in range(n_batches):
        y, X, y_len, X_len, dayIdx = next(iter(train_loader))

        out = model.train_one_epoch(X=X, y=y, X_len=X_len, y_len=y_len, dayIdx=dayIdx)
        loss = out["loss"]

        print(f"Batch no. {i}, Loss: {loss.item()}\n")
        if loss < min_loss:
            model_file = model_dir / "modelWeights"
            print(f"Save model state dict to: {model_file}")
            model.save_weights(model_file)
            min_loss = loss.item()


if __name__ == "__main__":
    args = {}
    args["seed"] = 0
    args["device"] = "cpu"
    args["dataset_path"] = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 64
    args["n_batches"] = 10000
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    # args["bidirectional"] = True
    main(args)
