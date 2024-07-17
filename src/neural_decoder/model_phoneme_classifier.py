import pickle
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# # 32 * 16 * 16
# class PhonemeClassifier(nn.Module):
#     def __init__(self, n_classes: int):
#         super(PhonemeClassifier, self).__init__()
#         # self.model =
#         self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(256 * 2 * 2, 512)
#         self.fc2 = nn.Linear(512, n_classes)

#     def forward(self, x):
#         # output = self.model(X)
#         # return oputput
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 256 * 2 * 2)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class PhonemeClassifier(nn.Module):
    def __init__(self, n_classes: int, input_shape: Tuple[int, ...]):
        super(PhonemeClassifier, self).__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x_reshaped = x.view(-1, *self.input_shape)
        return self.model(x_reshaped)


class PhonemeClassifierRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, num_layers: int = 1):
        super(PhonemeClassifierRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)  # * 2 for bidirectional

    def forward(self, x):
        # Assuming x is of shape (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(
            x.device
        )  # * 2 for bidirectional

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc(out)
        return out
