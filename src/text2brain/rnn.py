import torch
import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        # self.inputLayerNonlinearity = torch.nn.Softsign()
        # self.unfolder = torch.nn.Unfold(
        #     (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        # )
        # self.gaussianSmoother = GaussianSmoothing(
        #     neural_dim, 20, self.gaussianSmoothWidth, dim=1
        # )
        # self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        # self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        # for x in range(nDays):
        #     self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        self.embedding = nn.Embedding(n_classes + 1, neural_dim)  # +1 for CTC blank

        # GRU layers
        self.gru_decoder = nn.GRU(
            neural_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, neural_dim)
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, neural_dim)

    def forward(self, phoneme_indices):
        embedded = self.embedding(phoneme_indices)

        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                embedded.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                embedded.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(embedded, h0.detach())

        # get seq
        seq_out = self.fc_encoder_out(hid)
        return seq_out
