import pickle
from pathlib import Path
from types import SimpleNamespace
import math
from typing import List, Optional, Tuple, TypeVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from data.dataset import SyntheticPhonemeDataset
from neural_decoder.phoneme_utils import ROOT_DIR
from text2brain.models.model_interface import T2BGenInterface
from text2brain.models.vae_interface import VAEBase
from text2brain.visualization import plot_single_image
from utils import load_args


def logvar_to_std(logvar: torch.Tensor) -> torch.Tensor:
    """Convert the logarithm of the variance (logvar) to the standard deviation."""
    return torch.exp(0.5 * logvar)


class HierarchicalCondVAE(VAEBase, T2BGenInterface):
    def __init__(
        self, 
        latent_dims: List[int], 
        input_shape: Tuple[int, int, int], 
        classes: list, 
        conditioning: Literal["concat", "film"], 
        device: str = "cpu", 
        dec_emb_dim: int = None, 
        n_layers_film: int = None, 
        dec_hidden_dim: int = 512,
    ):
        super(HierarchicalCondVAE, self).__init__(0, input_shape, classes, device)
        self.device = device
        self.latent_dims = latent_dims
        self.classes = classes
        self.conditioning = conditioning

        if input_shape == (4, 64, 32):
            self.encoder = HierarchicalEncoder_4_64_32(latent_dims).to(device)
            self.decoder = HierarchicalDecoder_4_64_32(latent_dims, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim).to(device)
        else:
            raise ValueError(
                f"Invalid input shape ({input_shape}), we don't have a VAE version for this yet."
            )

    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, y)

    def reparameterize(self, mu_list: List[torch.Tensor], logvar_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sampling using the reparametrization trick."""
        z_list = []
        for mu, logvar in zip(mu_list, logvar_list):
            std = logvar_to_std(logvar)
            eps = torch.randn_like(std)
            z_list.append(mu + eps * std)
        return z_list

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        mu_list, logvar_list = self.encode(x)
        z_list = self.reparameterize(mu_list, logvar_list)
        recon_x = self.decode(z_list, y)
        return recon_x, mu_list, logvar_list

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        z_list = [torch.randn(y.size(0), ld).to(y.device) for ld in self.latent_dims]
        return self.decoder(z_list, y)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path):
        args = load_args(args_path)
        model = cls(
            latent_dims=args["latent_dims"], 
            input_shape=args["input_shape"], 
            classes=args["phoneme_cls"], 
            conditioning=args["conditioning"], 
            device=args["device"],
            dec_emb_dim=args.get("dec_emb_dim"),
            n_layers_film=args.get("n_layers_film"), 
            dec_hidden_dim=args.get("dec_hidden_dim", 512)
        )
        model.load_state_dict(torch.load(weights_path))
        return model

    def create_synthetic_phoneme_dataset(self, n_samples, neural_window_shape: Tuple[int, int, int]):
        pass


class HierarchicalEncoder_4_64_32(nn.Module):
    def __init__(self, latent_dims):
        super(HierarchicalEncoder_4_64_32, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Multiple latent dimensions for hierarchical encoding
        self.fc_mu = nn.ModuleList([nn.Linear(512 * 2 * 1, ld) for ld in latent_dims])
        self.fc_logvar = nn.ModuleList([nn.Linear(512 * 2 * 1, ld) for ld in latent_dims])

    def forward(self, x) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = x.view(-1, 4, 64, 32)
        h = self.model(x)
        h = h.view(h.size(0), -1)
        
        mu_list = [fc(h) for fc in self.fc_mu]
        logvar_list = [fc(h) for fc in self.fc_logvar]
        
        return mu_list, logvar_list

class HierarchicalDecoder_4_64_32(nn.Module):
    def __init__(self, latent_dims, classes: Optional[list] = None, dec_emb_dim: int = None, conditioning: Literal["concat", "film"] = None, n_layers_film: int = None, dec_hidden_dim: int = 512):
        super(HierarchicalDecoder_4_64_32, self).__init__()
        
        self.decoders = nn.ModuleList()
        for ld in latent_dims:
            self.decoders.append(
                nn.Sequential(
                    nn.Linear(ld, dec_hidden_dim * 2 * 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(dec_hidden_dim, 256, 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 16, 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(16, 4, 3, 1, 1, bias=False),
                    nn.Tanh(),
                )
            )

    def forward(self, z_list: List[torch.Tensor], labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = z_list[-1]  # Start with the most abstract latent variable
        
        for decoder in self.decoders:
            h = decoder(h.view(h.size(0), -1))
            h = h.view(h.size(0), -1, 2, 1)  # Adjust this reshape based on your architecture
        
        return h.view(-1, 1, 256, 32)



