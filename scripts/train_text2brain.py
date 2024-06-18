from neural_decoder.neural_decoder_trainer import getDatasetLoaders, trainModel
from text2brain.rnn import Text2BrainInterface
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, src):
        if len(src.size()) == 2:
            print(f"\nlen(src.size()) = {len(src.size())}")
            src = F.one_hot(src.long(), num_classes=41).float()
        print(f"src.size() = {src.size()}")
        outputs, hidden = self.gru(src)
        print(f"hidden.size() = {hidden.size()}")
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, hidden):
        batch_size = hidden.size(1)
        seq_len = 500
        inputs = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        outputs, hidden = self.gru(inputs, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden


class Seq2Seq(Text2BrainInterface):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)
        
    def forward(self, src):
        hidden = self.encoder(src)
        outputs, hidden = self.decoder(hidden)
        return outputs


class RNNText2Brain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1, bidirectional=False):
        super(RNNText2Brain, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional

        # GRU layers
        self.gru_encoder = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.gru_decoder = nn.GRU( 
            hidden_dim, 
            output_dim,
            batch_first=True
        )
        # self.fc = nn.Linear(hidden_dim, output_dim)

         # rnn outputs
        self.fc_decoder_out = nn.Linear(hidden_dim, output_dim) 

        
    def forward(self, text):
        # apply RNN layer
        print(f"\ntext.size() = {text.size()}")
        lengths = (text != 0).sum(dim=1)
        max_length = lengths.max().item()
        text = text[:, :max_length]
        print(f"text.size() = {text.size()}")

        if len(text.size()) == 2:
            text = F.one_hot(text.long(), num_classes=41).float()

        print(f"text.size() = {text.size()}")
        # h0 = torch.zeros(
        #     self.layer_dim,
        #     text.size(0),
        #     self.hidden_dim,
        # ).requires_grad_()
        # print(f"h0.size() = {h0.size()}")

        out, hidden = self.gru_encoder(text)
        print(f"out.size() = {out.size()}")
        print(f"hidden.size() = {hidden.size()}")
        out, hidden = self.gru_decoder(out,hidden)
        print(f"out.size() = {out.size()}")
        print(f"hidden.size() = {hidden.size()}")

        # get seq
        seq_out = self.fc_decoder_out(out)
        print(f"seq_out.size() = {seq_out.size()}")
        return seq_out




def main(args: dict) -> None:
    print("In main...")
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

    model = RNNText2Brain(
        input_dim=args["n_input_features"],
        output_dim=args["n_output_features"],
        hidden_dim=args["hidden_dim"],
        # layer_dim=2,
    )
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    for i in range(n_batches):
        model.train()
        y, X, y_len, X_len, dayIdx = next(iter(train_loader))

        output = model(X)
        print(f"\ny.size() = {y.size()}")
        a = y.size(1)
        b = output.size(1)
        print(f"output.size() = {output.size()}")
        
        print(f"a/b = {a/b}")



if __name__ == "__main__":
    args = {}
    args["seed"] = 0
    args["device"] = "cpu"
    args["dataset_path"] = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    args["batch_size"] = 2
    args["n_batches"] = 10
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 128
    args["n_layers"] = 3
    args["bidirectional"] = True
    main(args)