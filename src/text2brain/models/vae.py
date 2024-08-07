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

TypeCondVAE = TypeVar("TypeCondVAE", bound="CondVAE")
TypeVAE = TypeVar("TypeVAE", bound="VAE")


class DecoderInterface(nn.Module):
    def __init__(self, latent_dim: int, classes: Optional[list] = None, dec_emb_dim: int = None, conditioning: Literal["concat", "film"] = None, n_layers_film: int = None, dec_hidden_dim: int = 512):
        super(DecoderInterface, self).__init__()
        self.classes = classes if classes is not None else []
        self.dec_emb_dim = len(self.classes) if dec_emb_dim is None else dec_emb_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.conditioning = conditioning

    def _label_to_indices(self, labels, classes: List[int]):
        indices = []
        classes_torch = torch.tensor(classes).to(labels.device)
        for label in labels:
            index = (classes_torch == label).nonzero(as_tuple=True)[0].item()
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long, device=labels.device).view(labels.size())



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


class Decoder_1_256_32(DecoderInterface):
    def __init__(self, latent_dim: int, classes: Optional[list] = None, dec_emb_dim: int = None, conditioning: Literal["concat", "film"] = None, n_layers_film: int = None, dec_hidden_dim: int = 512):
        super(Decoder_1_256_32, self).__init__(latent_dim, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim)

        input_dim = latent_dim
        if classes is not None and self.conditioning == "concat":
            self.embedding = nn.Embedding(len(classes), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
        if self.conditioning == "film":
            self.film = FiLM(conditioning_dim=len(classes), in_features=latent_dim, n_layers=n_layers_film)

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

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if self.conditioning is not None:
            assert z.size(0) == labels.size(0)
            y_indices = self._label_to_indices(labels, self.classes)
            
            if self.conditioning == "concat":
                y_emb = self.embedding(y_indices)
                h = torch.concat((z, y_emb), dim=1)
            elif self.conditioning == "film":
                y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
                h = self.film(z, y_one_hot)
        else:
            h = z

        return self.decoder(h)


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

    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, 4, 64, 32)
        assert x[0].size() == (4, 64, 32)
        h = self.model(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Decoder for output size (4, 64, 32)
class Decoder_4_64_32(DecoderInterface):
    def __init__(self, latent_dim: int, classes: Optional[list] = None, dec_emb_dim: int = None, conditioning: Literal["concat", "film"] = None, n_layers_film: int = None, dec_hidden_dim: int = 512):
        super(Decoder_4_64_32, self).__init__(latent_dim, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim)
        
        input_dim = latent_dim
        if classes is not None and self.conditioning == "concat":
            self.embedding = nn.Embedding(len(classes), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
        if self.conditioning == "film":
            self.film = FiLM(conditioning_dim=len(classes), in_features=latent_dim, n_layers=n_layers_film)

        self.fc = nn.Sequential(nn.Linear(latent_dim, dec_hidden_dim * 2 * 1), nn.ReLU(inplace=True))  #TODO why 512*2
        self.model = nn.Sequential(
            nn.ConvTranspose2d(dec_hidden_dim, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 4 x 2
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

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None):

        if self.conditioning is not None:
            assert z.size(0) == labels.size(0)
            y_indices = self._label_to_indices(labels, self.classes)
            
            if self.conditioning == "concat":
                y_emb = self.embedding(y_indices)
                h = torch.concat((z, y_emb), dim=1)
            elif self.conditioning == "film":
                y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
                h = self.film(z, y_one_hot)
        else:
            h = z
        
        h = self.fc(h)
        h = h.view(h.size(0), self.dec_hidden_dim, 2, 1)  # reshape to (dec_hidden_dim) x 2 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h


# convolutions only over the channels. Input size (128, 8, 8)
class Encoder_128_8_8(nn.Module):
    def __init__(self, latent_dim: int):  # , classes: Optional[list] = None):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 128, 8, 8)
        assert x[0].size() == (128, 8, 8)
        h = self.model(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    
class Decoder_128_8_8(DecoderInterface):
    def __init__(self, latent_dim: int, classes: Optional[list] = None, dec_emb_dim: int = None, conditioning: Literal["concat", "film"] = None, n_layers_film: int = None, dec_hidden_dim: int = 512):
        super(Decoder_128_8_8, self).__init__(latent_dim, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim)

        input_dim = latent_dim
        if classes is not None and conditioning == "concat":
            self.embedding = nn.Embedding(len(classes), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
        if conditioning == "film":
            self.film = FiLM(conditioning_dim=len(classes), in_features=latent_dim, n_layers=n_layers_film)

        self.fc = nn.Sequential(nn.Linear(input_dim, dec_hidden_dim), nn.LeakyReLU(0.2, inplace=True))  # -> (512)
        self.model = nn.Sequential(
            # input is (512) x 1 x 1
            nn.ConvTranspose2d(dec_hidden_dim, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 2 x 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),  # -> (128) x 4 x 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),  # -> (64) x 8 x 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, 3, 1, 1, bias=False),  # -> (output_channels) x 8 x 8
            nn.Tanh(),  # TODO: change to softsign
        )

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass. Label should be not the index but the original phoneme class, such as in range(1, 40)"""
        
        if self.conditioning is not None:
            assert z.size(0) == labels.size(0)
            y_indices = self._label_to_indices(labels, self.classes)
            
            if self.conditioning == "concat":
                y_emb = self.embedding(y_indices)
                h = torch.concat((z, y_emb), dim=1)
            elif self.conditioning == "film":
                y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
                h = self.film(z, y_one_hot)
        else:
            h = z

        h = self.fc(h)
        h = h.view(h.size(0), self.dec_hidden_dim, 1, 1)
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h


class FiLM(nn.Module):
    def __init__(self, in_features: int, conditioning_dim: int, n_layers: int):
        super(FiLM, self).__init__()

        if n_layers not in [1, 2]:
            raise ValueError(f"n_layers for FiLM must be either 1 or 2, given: {n_layers}")
        
        if n_layers == 1:
            self.scale_network = nn.Sequential(
                nn.Linear(conditioning_dim, in_features)
            )
            self.shift_network = nn.Sequential(
                nn.Linear(conditioning_dim, in_features)
            )
        elif n_layers == 2:
            self.scale_network = nn.Sequential(
                nn.Linear(conditioning_dim, int(in_features / 2)),
                nn.ReLU(),
                nn.Linear(int(in_features / 2), in_features)
            )
            self.shift_network = nn.Sequential(
                nn.Linear(conditioning_dim, int(in_features / 2)),
                nn.ReLU(),
                nn.Linear(int(in_features / 2), in_features)
            )
    
    def forward(self, x, c):
        gamma = self.scale_network(c)
        beta = self.shift_network(c)

        return gamma * x + beta


class CondVAE(VAEBase, T2BGenInterface):
    def __init__(
        self, latent_dim: int, input_shape: Tuple[int, int, int], classes: list, conditioning: Literal["concat", "film"], device: str = "cpu", dec_emb_dim: int = None, n_layers_film: int = None, dec_hidden_dim: int = 512
    ):
        super(CondVAE, self).__init__(latent_dim, input_shape, classes, device)
        input_shape = tuple(input_shape)
        self.conditioning = conditioning

        # TODO give classes to encoder and decoder
        if input_shape == (1, 256, 32):
            self.encoder = Encoder_1_256_32(latent_dim).to(device)
            self.decoder = Decoder_1_256_32(latent_dim).to(device)
        elif input_shape == (4, 64, 32):
            self.encoder = Encoder_4_64_32(latent_dim).to(device)
            self.decoder = Decoder_4_64_32(latent_dim, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim).to(device)
        elif input_shape == (128, 8, 8):
            self.encoder = Encoder_128_8_8(latent_dim).to(device)
            self.decoder = Decoder_128_8_8(latent_dim, classes, dec_emb_dim, conditioning, n_layers_film, dec_hidden_dim).to(device)
        else:
            raise ValueError(
                f"Invalid input shape ({input_shape}), we don't have a VAE version for this yet."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, y)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sampling using the reparametrization trick."""
        std = logvar_to_std(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path) -> TypeCondVAE:
        args = load_args(args_path)
        dec_emb_dim = None if "dec_emb_dim" not in args.keys() else args["dec_emb_dim"]

        model = cls(
            latent_dim=args["latent_dim"], 
            input_shape=args["input_shape"], 
            classes=args["phoneme_cls"], 
            conditioning=args["conditioning"], 
            device=args["device"],
            dec_emb_dim=dec_emb_dim,
            n_layers_film=args["n_layers_film"], 
            dec_hidden_dim=args["dec_hidden_dim"]
        )
        model.load_state_dict(torch.load(weights_path))

        return model

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Generate an image from the latent space distribution.""" 
        assert len(y.size()) == 1

        z = torch.randn(y.size(0), self.latent_dim).to(y.device)
        return self.decoder(z, y)
    
    def create_synthetic_phoneme_dataset(self, n_samples, neural_window_shape: Tuple[int, int, int]) -> SyntheticPhonemeDataset:
        
        assert isinstance(neural_window_shape, tuple)

        classes = self.classes
        n_per_class = math.ceil(n_samples / len(classes))

        neural_windows = []
        phoneme_labels = []

        for label in classes:
            # label = self.decoder._label_to_indices(labels=[kls], classes=classes)
            label = torch.tensor([label]).to(self.device)
            
            for _ in range(n_per_class):
                neural_window = self.sample(y=label)
                neural_windows.append(neural_window.to("cpu"))
                phoneme_labels.append(label.to("cpu"))

        return SyntheticPhonemeDataset(neural_windows[:n_samples], phoneme_labels[:n_samples])



class VAE(VAEBase, T2BGenInterface):
    def __init__(self, latent_dim: int, input_shape: Tuple[int, int, int], device: str = "cpu", classes: List[int] = None, dec_hidden_dim: int = 512):
        super(VAE, self).__init__(latent_dim, input_shape, classes, device)
        input_shape = tuple(input_shape)

        if input_shape == (1, 256, 32):
            self.encoder = Encoder_1_256_32(latent_dim).to(device)
            self.decoder = Decoder_1_256_32(latent_dim).to(device)
        elif input_shape == (4, 64, 32):
            self.encoder = Encoder_4_64_32(latent_dim).to(device)
            self.decoder = Decoder_4_64_32(latent_dim).to(device)
        elif input_shape == (128, 8, 8):
            self.encoder = Encoder_128_8_8(latent_dim).to(device)
            self.decoder = Decoder_128_8_8(latent_dim, classes=classes, conditioning=None, dec_hidden_dim=dec_hidden_dim).to(device)
        else:
            raise ValueError(
                f"Invalid input shape ({input_shape}), we don't have a VAE version for this yet."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sampling using the reparametrization trick."""
        std = logvar_to_std(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate an image from the latent space distribution.""" 
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        return self.decoder(z)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path) -> TypeVAE:
        args = load_args(args_path)
        model = cls(
            latent_dim=args["latent_dim"], 
            input_shape=tuple(args["input_shape"]), 
            device=args["device"], 
            classes=args["phoneme_cls"], 
            dec_hidden_dim=args["dec_hidden_dim"],
        )
        model.load_state_dict(torch.load(weights_path))

        return model

    def create_synthetic_phoneme_dataset(self, n_samples, neural_window_shape: Tuple[int, int, int]):
        assert isinstance(neural_window_shape, tuple)

        classes = self.classes
        assert isinstance(classes, list)
        assert len(classes) == 1

        label = torch.tensor([classes[-1]]).to(self.device)
        neural_windows = []
        phoneme_labels = []

        for i in range(n_samples):
            neural_window = self.sample(n_samples=1).view(1, *neural_window_shape)
            neural_windows.append(neural_window.to("cpu"))
            phoneme_labels.append(label.to("cpu"))
            # if i < 20:
            #     plot_single_image(
            #         X=neural_window.squeeze().cpu().detach().numpy(), 
            #         out_file=ROOT_DIR / "evaluation" / "synthetic_phoneme_dataset" / f"img_{i}_cls_{self.classes[-1]}.png",
            #         title=f"Synthetic phoneme image (phoneme cls {self.classes[-1]})"
            #     )

        return SyntheticPhonemeDataset(neural_windows, phoneme_labels)


def logvar_to_std(logvar: torch.Tensor) -> torch.Tensor:
    """Convert the logarithm of the variance (logvar) to the standard deviation."""
    return torch.exp(0.5 * logvar)


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
