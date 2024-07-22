from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(VAE, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim

#         # encoder
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # mean
#         self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # log-variance

#         # decoder
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Encodes the input into latent space.
#         """
#         x = x.view(-1, self.input_dim)
#         h1 = torch.relu(self.fc1(x))
#         mu = self.fc2_mu(h1)
#         logvar = self.fc2_logvar(h1)
#         return mu, logvar

#     def decode(self, z) -> torch.Tensor:
#         """
#         Decodes the latent space into reconstructed input.
#         """
#         h3 = torch.relu(self.fc3(z))
#         out = torch.sigmoid(self.fc4(h3))
#         return out

#     def draw_sample(self, mu, logvar):
#         """
#         Draws a sample from the latent space using reparameterization trick.
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         """
#         Forward pass through the VAE.
#         """
#         mu, logvar = self.encode(x)
#         z = self.draw_sample(mu, logvar)
#         out = self.decode(z)
#         out = out.view(x.size())
#         return out, mu, logvar


# # convolutions only over the channels. Input size (1, 256, 32)
# class Encoder(nn.Module):
#     def __init__(self, input_channels, latent_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
#             nn.ReLU(),
#             nn.Conv2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(256, 256),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(256, latent_dim)
#         self.fc_logvar = nn.Linear(256, latent_dim)

#     def forward(self, x):
#         assert x[0].size() == (1, 256, 32)
#         x = x.view(-1, 1, 256, 32)
#         h = self.encoder(x)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar


# class Decoder(nn.Module):
#     def __init__(self, latent_dim, output_channels):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Unflatten(1, (1, 256, 1)),
#             nn.ConvTranspose2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, output_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         return self.decoder(z)


# convolutions only over the channels. Input size (128, 8, 8)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
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
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 1, 1)  # reshape to (512) x 1 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h


class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = Encoder(latent_dim)

        # Decoder
        self.decoder = Decoder(latent_dim)

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


def compute_kl_divergence(logvar, mu):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class ELBOLoss(nn.Module):
    def __init__(self, mse_reduction: str):
        super(ELBOLoss, self).__init__()
        self.mse_reduction = mse_reduction

    def forward(self, reconstructed_x, x, mu, logvar):
        mse = F.mse_loss(reconstructed_x, x, reduction=self.mse_reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu)
        return mse, kld


# implementation from: https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
class GECOLoss(nn.Module):
    def __init__(
        self, goal, step_size, mse_reduction: str, alpha=0.99, beta_init=1.0, beta_min=1e-10, speedup=None
    ):
        super(GECOLoss, self).__init__()
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.mse_reduction = mse_reduction
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
        mse = F.mse_loss(reconstructed_x, x, reduction=self.mse_reduction)
        kld = compute_kl_divergence(logvar=logvar, mu=mu)
        loss = self.compute_contrained_loss(err=mse, kld=kld)

        return loss
