import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from data.dataset import SyntheticPhonemeDataset
from text2brain.models.conditioning_film import FiLM
from text2brain.models.model_interface import T2BGenInterface
from text2brain.models.utils import labels_to_indices
from utils import load_args


def phonemes_to_signal(model, phonemes: list, signal_shape: Tuple[int, ...] = (32, 16, 16)) -> torch.Tensor:
    """
    Given a list of phonemes, generate a signal for each and concatenate them
    """
    # assert model.conditional == True

    unique_phonemes = list(set(phonemes))
    assert all(p in model.classes for p in unique_phonemes)

    signal = []
    for p in phonemes:
        s = model.generate(label=torch.tensor([p]))
        s = s.view(1, *signal_shape)
        signal.append(s)

    return torch.cat(signal, dim=1)


def _get_indices_in_classes(labels: torch.Tensor, classes: torch.Tensor):
    indices = []
    for label in labels:
        index = torch.where(classes == label)[0]
        indices.append(index)
        if not index.numel() > 0:
            raise ValueError(f"Invalid label given: {label}")

    return torch.tensor(indices).int()


class PhonemeImageGAN(T2BGenInterface, nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        classes: List[int],
        conditioning: Literal["film"], 
        device: str,
        lr_g: float,
        lr_d: float,
        dec_emb_dim: int = None,
        n_critic: int = 5,
    ):
        super(PhonemeImageGAN, self).__init__()
        self.lambda_gp = 10
        self.conditioning = conditioning
        
        self.classes = classes
        self.n_classes = len(self.classes)
        self.input_shape = tuple(input_shape)

        if self.input_shape == (4, 64, 32):
            self._g = Generator_4_64_32(latent_dim, classes, conditioning=conditioning, dec_emb_dim=dec_emb_dim).to(device)
            self._d = Discriminator_4_64_32(128, classes, conditioning=conditioning).to(device)  # TODO
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")
        
        self.device = device
        self.n_critic = n_critic
        self.lr_g = lr_g
        self.lr_d = lr_d

        self._g.apply(self._weights_init)
        self._d.apply(self._weights_init)

        self.init_optimizers()

    def init_optimizers(self) -> None:
        self.optim_d = optim.Adam(self._d.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self._g.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        # random weight term for interpolation between real and fake samples
        alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=self.device).expand_as(real_samples)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self._d(interpolates, y)

        # compute gradients w.r.t. interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2))

        return gradient_penalty


    def train_step(self, dl):
        # update discriminator
        losses_D = []
        for _ in range(self.n_critic):
            self._d.zero_grad()

            X_real, y, _, _ = next(iter(dl))
            X_real = X_real.to(self.device)
            y = y.to(self.device)
            output_real = self._d(X_real, y)

            noise = torch.randn(y.size(0), self._g.latent_dim, device=self.device)
            X_fake = self._g(noise, y)
            output_fake = self._d(X_fake.detach(), y)

            loss_D = -torch.mean(output_real) + torch.mean(output_fake)

            gradient_penalty = self.compute_gradient_penalty(X_real, X_fake.detach(), y)
            loss_D += self.lambda_gp * gradient_penalty

            loss_D.backward()


            # grad_norms = []
            # for p in self._d.parameters():
            #     if p.grad is not None:
            #         grad_norms.append(p.grad.norm(2).item())
            # print(f"Discriminator Gradient Norms: {grad_norms}")
            
            
            # grad_norms = []
            # for p in self._d.parameters():
            #     if p.grad is not None:
            #         grad_norms.append(p.grad.norm(2).item())
            # print(f"Discriminator Gradient Norms: {grad_norms}")
            
            self.optim_d.step()
            losses_D.append(loss_D)

        loss_D = sum(losses_D) / len(losses_D)

        # update generator
        X_real, y, _, _ = next(iter(dl))
        X_real = X_real.to(self.device)
        y = y.to(self.device)

        self._g.zero_grad()
        noise = torch.randn(y.size(0), self._g.latent_dim, device=self.device)
        X_fake = self._g(noise, y)
        output_fake = self._d(X_fake, y)

        loss_G = -torch.mean(output_fake)
        loss_G.backward()

        self.optim_g.step()

        return loss_D, loss_G

    def generate(self, label: torch.Tensor = None):
        """Generate an image. Label is the class label, not the index."""
        noise = torch.randn(1, self._g.latent_dim, device=self.device)
        return self.generate_from_given_noise(noise=noise, label=label)

    def generate_from_given_noise(self, noise: torch.Tensor, label: torch.Tensor = None):
        """Generate an image. Label is the class label, not the index."""
        self._g.eval()
        assert noise.size(0) == 1
        assert noise.size(1) == self._g.latent_dim
        
        gen_img = self._g(noise, label)
        return gen_img


    def create_synthetic_phoneme_dataset(self, n_samples: int, neural_window_shape: Tuple[int, int, int] = (1, 256, 32)) -> SyntheticPhonemeDataset:
        
        assert isinstance(neural_window_shape, tuple)

        classes = self.classes
        n_per_class = math.ceil(n_samples / len(classes))

        neural_windows = []
        phoneme_labels = []

        with torch.no_grad():
            for label in classes:
                label = torch.tensor([label]).to(self.device)

                for _ in range(n_per_class):
                    neural_window = self.generate(label=label)
                    neural_window = neural_window.view(neural_window.size(0), *neural_window_shape)

                    neural_windows.append(neural_window.to("cpu"))
                    phoneme_labels.append(label.to("cpu"))

        return SyntheticPhonemeDataset(neural_windows[:n_samples], phoneme_labels[:n_samples])

    def save_state_dict(self, path: str):
        print(f"Store model state dict to: {path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path):
        args = load_args(args_path)
        if "dec_emb_dim" in args.keys():
            dec_emb_dim = args["dec_emb_dim"]
        else:
            dec_emb_dim = None
    
        model = cls(
            input_shape=tuple(args["input_shape"]),
            latent_dim=args["latent_dim"],
            classes=args["phoneme_cls"],
            conditioning=args["conditioning"], 
            dec_emb_dim=dec_emb_dim,
            device=args["device"],
            lr_g=args["lr_g"],
            lr_d=args["lr_d"],
            n_critic=args["n_critic"],
        )
        model.load_state_dict(torch.load(weights_path))

        return model


class Generator_128_8_8(nn.Module):
    def __init__(self, latent_dim: int, classes: List[int], dec_emb_dim: int = 32):
        super(Generator_128_8_8, self).__init__()

        self.input_shape = (128, 8, 8)
        self.classes = classes
        self.n_classes = len(classes)

        self.dec_emb_dim = dec_emb_dim
        self.latent_dim = latent_dim
        input_dim = latent_dim
        
        if len(classes) > 1:
            self.conditional = True
            self.embedding = nn.Embedding(len(classes), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
        else:
            self.conditional = False
            self.embedding = None
 
        self.fc = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(inplace=True))  # -> (512)
        self.model = nn.Sequential(
            # input is (512) x 1 x 1
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 2 x 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),  # -> (128) x 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),  # -> (64) x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 3, 1, 1, bias=False),  # -> (output_channels) x 8 x 8
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass of the generator. Takes y as indices indices, not class labels. """
        
        if self.conditional:
            y_emb = self.embedding(y)
            h = torch.concat((noise, y_emb), dim=1)
            # y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
        else:
            h = noise

        h = self.fc(h)
        h = h.view(h.size(0), 256, 1, 1)
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h

    
class Generator_4_64_32(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        classes: List[int], 
        conditioning: Optional[Literal["film"]], 
        dec_emb_dim: int = None, 
        n_layers_film: int = 2
    ):
        super(Generator_4_64_32, self).__init__()

        self.input_shape = (4, 64, 32)
        self.classes = classes
        self.n_classes = len(classes)
        # if self.n_classes == 1:
        #     assert conditioning is None, f"Only one class is present ({classes}), but conditioning is not None: {conditioning}"
        
        self.conditioning = conditioning
        self.dec_emb_dim = dec_emb_dim
        self.latent_dim = latent_dim
        self.input_dim = latent_dim

        if self.conditioning == "concat":
            assert dec_emb_dim is not None

            self.embedding = nn.Embedding(len(classes), self.dec_emb_dim)
            self.input_dim += self.dec_emb_dim
        if self.conditioning == "film":
            self.film = FiLM(conditioning_dim=len(classes), in_features=latent_dim, n_layers=n_layers_film)

        dec_hidden_dim = 512
        self.dec_hidden_dim = dec_hidden_dim

        self.fc = nn.Sequential(nn.Linear(self.input_dim, dec_hidden_dim * 2 * 1), nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(
            nn.ConvTranspose2d(dec_hidden_dim, 256, 3, 2, 1, 1, bias=False),  # -> (256) x 4 x 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False),  # -> (128) x 8 x 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),  # -> (64) x 16 x 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=False),  # -> (32) x 32 x 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1, bias=False),  # -> (16) x 64 x 32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 4, 3, 1, 1, bias=False),  # -> (4) x 64 x 32
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass of the generator. Takes y as indices indices, not class labels. """
        
        if self.conditioning is not None: 
            assert noise.size(0) == labels.size(0)
            y_indices = labels_to_indices(labels, self.classes)

            if self.conditioning == "concat":
                y_emb = self.embedding(y_indices)
                h = torch.concat((noise, y_emb), dim=1)
            elif self.conditioning == "film":
                y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
                h = self.film(noise, y_one_hot)
        else:
            h = noise

        h = self.fc(h)
        h = h.view(h.size(0), self.dec_hidden_dim, 2, 1)  # reshape to (dec_hidden_dim) x 2 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h

class Discriminator_128_8_8(nn.Module):
    def __init__(self, n_channels: int, classes: List[int]):
        super(Discriminator_128_8_8, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.input_shape = (128, 8, 8)

        if len(classes) > 1:
            self.conditional = True
            input_dim = n_channels + self.n_classes
            self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        else:
            self.conditional = False
            input_dim = n_channels
            self.label_emb = None

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, 1, 1, bias=False),  # -> (64) x 8 x 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # -> (128) x 4 x 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (128) x 2 x 2
            nn.Conv2d(128, 256, 3, 1, 1),  # -> (256) x 2 x 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (256) x 1 x 1
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),  # -> (512)
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img: torch.Tensor, y: torch.Tensor):
        img = img.view(-1, *self.input_shape)
        if self.conditional:
            y = self.label_emb(y)
            y = y.view(y.size(0), y.size(1), 1, 1)
            y = y.repeat(1, 1, img.size(2), img.size(3))
            d_in = torch.cat((img, y), 1)
        else:
            d_in = img

        output = self.model(d_in)
        return output.view(-1)


class Discriminator_4_64_32(nn.Module):
    def __init__(
        self, 
        n_channels: int, 
        classes: List[int],
        conditioning: Optional[Literal["film"]],
    ):
        super(Discriminator_4_64_32, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.input_shape = (4, 64, 32)
        self.conditioning = conditioning

        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1, bias=False),  # -> (16) x 64 x 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> (32) x 32 x 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> (64) x 16 x 8
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 8 x 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # -> (64) x 4 x 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 2, 1, bias=False),  # -> (256) x 2 x 1
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.fc1 = nn.Linear(256 * 2 * 1, 512)
        self.fc2 = nn.Linear(512, 1)
        

    def forward(self, img: torch.Tensor, y: torch.Tensor):
        img = img.view(-1, *self.input_shape)
        d_in = img

        out = self.model(d_in)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

