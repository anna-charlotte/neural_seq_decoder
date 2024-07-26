import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import PhonemeDataset
from evaluation import compute_correlation_matrix, compute_likelihood
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import TransposeTransform
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


class DummyVAE(nn.Module):
    def __init__(self):
        super(DummyVAE, self).__init__()
        self.encoder = DummyEncoder()

    def forward(self, x):
        means, logvars = self.encoder(x)
        return x, means, logvars


class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.latent_dim = 10

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


test_compute_likelihood()

# def test_tanh_vs_softsign():
#     from torchvision import transforms
#     from neural_decoder.transforms import (
#         AddOneDimensionTransform,
#         ReorderChannelTransform,
#         SoftsignTransform,
#         TransposeTransform,
#     )
#     from data.augmentations import GaussianSmoothing

#     test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
#     with open(test_file, "rb") as handle:
#         data = pickle.load(handle)

#     transform = transforms.Compose(
#         [
#             TransposeTransform(0, 1),
#             ReorderChannelTransform(),
#             AddOneDimensionTransform(dim=0),
#             GaussianSmoothing(256, kernel_size=20, sigma=2.0, dim=1),
#         ]
#     )
#     plot_dir = ROOT_DIR / "evaluation" / "softsign_vs_tanh"
#     plot_dir.mkdir(parents=True, exist_ok=True)

#     dl = get_data_loader(
#         data=data,
#         batch_size=1,
#         shuffle=True,
#         collate_fn=None,
#         transform=transform,
#         dataset_cls=PhonemeDataset,
#     )
#     x = dl.dataset.neural_windows[0]
#     for i, batch in enumerate(dl):
#         X, _, _, _ = batch
#         X = X[0][0]
#         X_softsign = F.softsign(X)
#         X_tanh = torch.tanh(X)

#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#         # Plot each image
#         axes[0].imshow(X, cmap='plasma')
#         axes[0].axis('off')
#         axes[0].set_title('Original Image')

#         axes[1].imshow(X_softsign, cmap='plasma')
#         axes[1].axis('off')
#         axes[1].set_title('Softsign(image)')

#         axes[2].imshow(X_tanh, cmap='plasma')
#         axes[2].axis('off')
#         axes[2].set_title('Tanh(image)')

#         plt.tight_layout()
#         plt.savefig(plot_dir / f"softsign_vs_tanh_{i}.png")

#         if i == 10:
#             break
