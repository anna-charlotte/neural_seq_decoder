import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from text2brain.models.t2b_interface import TextToBrainInterface


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
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


class TextToBrainGRU(TextToBrainInterface):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(TextToBrainGRU, self).__init__()
        self.model = RNN(input_dim, hidden_dim, output_dim, n_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_len: torch.Tensor,
        y_len: torch.Tensor,
        dayIdx: torch.Tensor,
    ) -> dict:
        self.model.train()
        output = self.model(X)
        y_pred, y_true = pad_to_match(output, y)

        loss = self.criterion(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss}

    def predict(self, X) -> torch.Tensor:
        self.model.eval()
        return self.model(X)

    def save_weights(self, file_path: Path) -> None:
        torch.save(self.model.state_dict(), file_path)

    def load_weights(self, file_path: Path) -> None:
        self.model.load_state_dict(torch.load(file_path))


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


def load_model(model_dir: Path):
    args_file = Path(model_dir / "args")
    weights_file = Path(model_dir / "modelWeights")

    assert args_file.exists()
    assert weights_file.exists()

    with open(args_file, "rb") as handle:
        args = pickle.load(handle)

    if args["model_class"] == TextToBrainGRU.__name__:
        model = TextToBrainGRU(
            input_dim=args["n_input_features"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["n_output_features"],
            n_layers=args["n_layers"],
        )
    else:
        raise ValueError(f"model_class is not valid: {args['model_class']}")

    model.load_weights(file_path=weights_file)

    return model
