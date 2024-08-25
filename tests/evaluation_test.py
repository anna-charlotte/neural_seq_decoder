import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import PhonemeDataset
from evaluation import compute_correlation_matrix, compute_likelihood, compute_auroc_with_stderr, compute_man_whitney
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import TransposeTransform
from text2brain.models.vae_interface import VAEBase
from text2brain.visualization import plot_correlation_matrix


def test_compute_correlation_matrix():
    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        data = pickle.load(handle)

    transform = TransposeTransform(dim1=0, dim2=1)
    dl = get_data_loader(
        data=data,
        batch_size=128,
        shuffle=True,
        collate_fn=None,
        transform=transform,
        dataset_cls=PhonemeDataset,
    )

    neural_windows, phonemes, _, _ = next(iter(dl))
    for X, y in zip(neural_windows, phonemes):

        corr_matrix = compute_correlation_matrix(X.cpu().detach().numpy())

        assert corr_matrix.shape == (256, 256)
        assert not np.isnan(corr_matrix).any(), "The correlation matrix contains NaNs."
        assert np.all(
            (corr_matrix >= -1) & (corr_matrix <= 1)
        ), "The correlation matrix contains values outside the range [-1, 1]."


class DummyVAE(VAEBase):
    def __init__(self, latent_dim=10, input_shape=(3, 28, 28)):
        super(DummyVAE, self).__init__(latent_dim, input_shape)
        self.encoder = DummyEncoder(latent_dim)

    def forward(self, x):
        means, logvars = self.encoder(x)
        return x, means, logvars

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return z


class DummyEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(DummyEncoder, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        means = torch.zeros((x.size(0), self.latent_dim))
        logvars = torch.zeros((x.size(0), self.latent_dim))
        return means, logvars


def test_compute_likelihood():
    vae = DummyVAE()
    real_data = torch.rand((16, 3, 28, 28))  # batch of 16 RGB-images
    x_recon, means, logvars = vae(real_data)

    assert x_recon.size() == real_data.size()
    assert means.size() == (16, 10)
    assert logvars.size() == (16, 10)

    avg_mean = torch.mean(means, dim=0)
    avg_logvar = torch.mean(logvars, dim=0)

    assert avg_mean.size() == (10,)
    assert avg_logvar.size() == (10,)

    synthetic_images = torch.rand((2, 3, 28, 28))
    likelihoods = compute_likelihood(avg_mean, avg_logvar, synthetic_images, vae)
    assert isinstance(likelihoods, np.ndarray)
    assert likelihoods.size() == (2,)
    # average_likelihood = torch.mean(likelihoods)
    # print(f'Average Likelihood of Synthetic Data: {average_likelihood.item()}')


def test_compute_auroc_with_stderr():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1])
    mean, std = compute_auroc_with_stderr(y_true, y_pred, n_iters=100)

    print(f"mean = {mean}")
    print(f"std = {std}")

    mean, std = compute_auroc_with_stderr(y_true, y_true, n_iters=100)
    
    print(f"mean = {mean}")
    print(f"std = {std}")




def test_compute_man_whitney():
    y_true = np.array([0] * 1000 + [1] * 1000)
    assert len(y_true) == 2000
    stat, p_val = compute_man_whitney(y_true, y_true, y_true, n_iters=100)
    
    assert stat == 5000.0
    assert p_val == 1.0
test_compute_auroc_with_stderr()
test_compute_man_whitney()