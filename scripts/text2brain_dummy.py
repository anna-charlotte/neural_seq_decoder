import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_decoder.neural_decoder_trainer import getDatasetLoaders
from text2brain.rnn import load_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class TextToBrainGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(TextToBrainGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru_encoder = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gru_decoder = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward propagate GRU
        lengths = (x != 0).sum(dim=1)
        max_length = lengths.max().item()
        x = x[:, :max_length]

        if len(x.size()) == 2:
            x = F.one_hot(x.long(), num_classes=41).float()

        output, hidden = self.gru_encoder(x)
        # output size = (bs, n, 128)
        # hidden size = (n_layers=1, bs, 128)

        batch_size = x.size(0)

        # Prepare the initial input for the decoder (start token, could be zeros)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim)

        # artificially set the output length
        output_length = 12 * x.size(1)

        # Initialize the output tensor
        outputs = torch.zeros(batch_size, output_length, self.fc.out_features).to(
            x.device
        )

        # Decode the encoded context vector
        for t in range(output_length):
            decoder_output, hidden = self.gru_decoder(decoder_input, hidden)
            out = self.fc(decoder_output)
            outputs[:, t, :] = out.squeeze(1)
            decoder_input = decoder_output

        return outputs


def pad_to_match(tensor_a, tensor_b):
    """
    Pads the shorter tensor to match the longer one.
    Assumes both tensors have the same batch size and feature dimension.
    """
    if tensor_a.size(1) < tensor_b.size(1):
        padding_size = tensor_b.size(1) - tensor_a.size(1)
        padded_tensor = F.pad(tensor_a, (0, 0, 0, padding_size, 0, 0))
        return padded_tensor, tensor_b
    else:
        padding_size = tensor_a.size(1) - tensor_b.size(1)
        padded_tensor = F.pad(tensor_b, (0, 0, 0, padding_size, 0, 0))
        return tensor_a, padded_tensor


def main(args: dict) -> None:

    os.makedirs(args["output_dir"], exist_ok=True)

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = args["device"]
    n_batches = args["n_batches"]
    batch_size = args["batch_size"]

    print("Get data loaders ...")
    train_loader, test_loader, loaded_data = getDatasetLoaders(
        args["dataset_path"],
        batch_size,
    )

    # Parameters
    input_dim = args["n_input_features"]
    hidden_dim = args["hidden_dim"]
    output_dim = args["n_output_features"]
    n_layers = args["n_layers"]

    # Initialize the model, loss function, and optimizer
    model = TextToBrainGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
    )

    args["model_class"] = model.__class__.__name__

    model_dir = Path(args["output_dir"]) / "model"
    args_file = model_dir / "args"
    with open(args_file, "wb") as file:
        print(f"Write args to: {args_file}")
        pickle.dump(args, file)

    model = load_model(model_dir=model_dir)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = float("inf")

    for i in range(n_batches):
        model.train()
        y, X, y_len, X_len, dayIdx = next(iter(train_loader))

        output = model(X)
        y_pred, y_true = pad_to_match(output, y)

        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch no. {i}, Loss: {loss.item()}\n")
        if loss < min_loss:
            model_file = model_dir / "modelWeights"
            print(f"Save model state dict to: {model_file}")
            torch.save(model.state_dict(), model_file)
            min_loss = loss.item()


def store_synthetic_data():
    pass


if __name__ == "__main__":
    args = {}
    args["seed"] = 0
    args["device"] = "cpu"
    args["dataset_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    )
    args["output_dir"] = (
        "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    )
    args["batch_size"] = 64
    args["n_batches"] = 10000
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    # args["bidirectional"] = True
    main(args)
