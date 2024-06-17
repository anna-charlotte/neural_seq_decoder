from neural_decoder.neural_decoder_trainer import getDatasetLoaders, trainModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_layers,
        bidirectional=False,
    ):
        super(GRUEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.kernelLen = 14
        self.strideLen = 4
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
       
        self.gru_decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # self.inpLayer = nn.Linear(input_dim, input_dim)
        # self.inpLayer.weight = torch.nn.Parameter(self.inpLayer.weight + torch.eye(input_dim))

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, output_dim
            ) 
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, phoneme_seqs):
        print(f"phoneme_seqs.size() = {phoneme_seqs.size()}")
        if len(phoneme_seqs.size()) == 2:
            print(f"len(phoneme_seqs.size()) = {len(phoneme_seqs.size())}")
            phoneme_seqs = F.one_hot(phoneme_seqs.long(), num_classes=41).float()
            print(f"len(phoneme_seqs.size()) = {len(phoneme_seqs.size())}")
            print(f"phoneme_seqs.size() = {phoneme_seqs.size()}")

        # stride/kernel
        strided_inputs = phoneme_seqs
        # strided_inputs = self.unfolder(phoneme_seqs)
        # print(f"strided_inputs.size() = {strided_inputs.size()}")

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.n_layers * 2,
                phoneme_seqs.size(0),
                self.hidden_dim,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.n_layers,
                phoneme_seqs.size(0),
                self.hidden_dim,
            ).requires_grad_()

        hid, _ = self.gru_decoder(strided_inputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
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
        args["batch_size"],
    )

    model = GRUEncoder(
        input_dim=args["n_input_features"],
        output_dim=args["n_out_features"],
        hidden_dim=args["hidden_dim"],
        n_layers=args["n_layers"],
        bidirectional=args["bidirectional"],
    ).to(device)
    print(f"model = {model}")

    for i in range(n_batches):
        model.train()

        y, X, y_len, X_len, dayIdx = next(iter(train_loader))

        ouput = model(X)
        print(f"output.size() = {output.size()}")

if __name__ == "__main__":
    args = {}
    args["seed"] = 0
    args["device"] = "cpu"
    args["dataset_path"] = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    args["batch_size"] = 16
    args["n_batches"] = 10
    args["n_input_features"] = 41
    args["n_out_features"] = 256
    args["hidden_dim"] = 256
    args["n_layers"] = 3
    args["bidirectional"] = True
    main(args)