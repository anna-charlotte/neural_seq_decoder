import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from text2brain.models.vae_interface import VAEBase


# convolutions only over the channels. Input size (1, 256, 32)
class Encoder_1_256_32(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_1_256_32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        assert x[0].size() == (1, 256, 32)
        x = x.view(-1, 1, 256, 32)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder_1_256_32(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_1_256_32, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Unflatten(1, (1, 256, 1)),
            nn.ConvTranspose2d(
                1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)
            ),
            nn.Tanh(),  # TODO: change to softsign
        )

    def forward(self, z):
        return self.decoder(z)


# convolutions only over the channels. Input size (128, 8, 8)
class Encoder_128_8_8(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_128_8_8, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),  # -> (64) x 8 x 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 4 x 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # -> (256) x 2 x 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),  # -> (512) x 1 x 1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = x.view(-1, 128, 8, 8)
        assert x[0].size() == (128, 8, 8)
        h = self.model(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder_128_8_8(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_128_8_8, self).__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(inplace=True))  # -> (512)
        self.model = nn.Sequential(
            # input is (512) x 1 x 1
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 2 x 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),  # -> (128) x 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),  # -> (64) x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 3, 1, 1, bias=False),  # -> (output_channels) x 8 x 8
            nn.Tanh(),  # TODO: change to softsign
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 1, 1)  # reshape to (512) x 1 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h


class Encoder_4_64_32(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_4_64_32, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1, bias=False),  # -> (16) x 64 x 32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> (32) x 32 x 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> (64) x 16 x 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 8 x 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # -> (256) x 4 x 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),  # -> (512) x 2 x 1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(512 * 2 * 1, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 1, latent_dim)

    def forward(self, x):
        x = x.view(-1, 4, 64, 32)
        assert x[0].size() == (4, 64, 32)
        h = self.model(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Decoder for output size (4, 64, 32)
class Decoder_4_64_32(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_4_64_32, self).__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512 * 2 * 1), nn.ReLU(inplace=True))
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 4 x 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),  # -> (128) x 8 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),  # -> (64) x 16 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=False),  # -> (32) x 32 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1, bias=False),  # -> (16) x 64 x 32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, 3, 1, 1, bias=False),  # -> (4) x 64 x 32
            nn.Tanh(),  # TODO: change to softsign
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 2, 1)  # reshape to (512) x 2 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h


class VAE(VAEBase):
    def __init__(self, latent_dim, input_shape: Tuple[int, int, int], device: str = "cpu"):
        super(VAE, self).__init__(latent_dim, input_shape, device)

        if input_shape == (1, 256, 32):
            self.encoder = Encoder_1_256_32(latent_dim).to(device)
            self.decoder = Decoder_1_256_32(latent_dim).to(device)
        elif input_shape == (4, 64, 32):
            self.encoder = Encoder_4_64_32(latent_dim).to(device)
            self.decoder = Decoder_4_64_32(latent_dim).to(device)
        elif input_shape == (128, 8, 8):
            self.encoder = Encoder_128_8_8(latent_dim).to(device)
            self.decoder = Decoder_128_8_8(latent_dim).to(device)
        else:
            raise ValueError(
                f"Invalid input shape ({input_shape}), we don't have a VAE version for this yet."
            )

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Sampling using the reparametrization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def save_state_dict(self, path: str):
        print(f"Store model state dict to: {path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path):
        with open(args_path, "rb") as file:
            args = pickle.load(file)

        model = cls(latent_dim=args["latent_dim"], input_shape=args["input_shape"], device=args["device"])
        model.load_state_dict(torch.load(weights_path))

        return model


def compute_mean_logvar_mse(vae: VAEBase, dl: DataLoader) -> SimpleNamespace:
    assert dl.batch_size == 1, f"Expected a dataloader with batchsize 1, give: batch_size={dl.batch_size}"
    vae.eval()

    sum_means = torch.zeros(vae.latent_dim)
    sum_logvars = torch.zeros(vae.latent_dim)
    sum_mse = torch.tensor(0.0)

    for batch in dl:
        X, _, _, _ = batch
        X = X.to(vae.device)
        X_recon, mean, logvar = vae(X)
        mse = F.mse_loss(X_recon, X, reduction="mean")

        sum_means += mean.squeeze().cpu()
        sum_logvars += logvar.squeeze().cpu()
        sum_mse += mse.item()

    mean = sum_means / len(dl)
    logvar = sum_logvars / len(dl)
    mse = sum_mse / len(dl)

    return SimpleNamespace(mean=mean, logvar=logvar, mse=mse)


def get_sample(vae: VAEBase, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    z = vae.reparametrize(mu=mean, logvar=logvar)
    recon_x = vae.decode(z=z)
    return recon_x


def compute_kl_divergence(logvar, mu, reduction: str):
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif reduction == "mean":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif reduction == "none":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class ELBOLoss(nn.Module):
    def __init__(self, reduction: str):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction

    def forward(self, reconstructed_x, x, mu, logvar):
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        return mse, kld


# implementation from: https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
class GECOLoss(nn.Module):
    def __init__(
        self, goal, step_size, reduction: str, alpha=0.99, beta_init=1.0, beta_min=1e-10, speedup=None
    ):
        super(GECOLoss, self).__init__()
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.reduction = reduction
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.speedup = speedup

    def to(self, device: str):
        self.beta = self.beta.to(device)
        self.beta_min = self.beta_min.to(device)
        self.beta_max = self.beta_max.to(device)
        if self.err_ema is not None:
            self.err_ema = self.err_ema.to(device)

    def compute_contrained_loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld

        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema
            constraint = self.goal - self.err_ema

            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)

            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)

        return loss

    def forward(self, reconstructed_x, x, mu, logvar):
        mse = F.mse_loss(reconstructed_x, x, reduction=self.reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu, reduction=self.reduction)
        loss = self.compute_contrained_loss(err=mse, kld=kld)

        return loss
