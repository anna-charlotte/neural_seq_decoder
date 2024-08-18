import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from data.dataset import SyntheticPhonemeDataset
from text2brain.models.model_interface import T2BGenInterface
from utils import load_args


def phonemes_to_signal(model, phonemes: list, signal_shape: Tuple[int, ...] = (32, 16, 16)) -> torch.Tensor:
    """
    Given a list of phonemes, generate a signal for each and concatenate them
    """
    assert model.conditional == True

    unique_phonemes = list(set(phonemes))
    assert all(p in model.phoneme_cls for p in unique_phonemes)

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

# import torch
# import torch.autograd as autograd
# def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels):
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=real_samples.device)
#     alpha = alpha.expand_as(real_samples)

#     # Get random interpolation between real and fake samples
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     interpolates = interpolates.requires_grad_(True)

#     d_interpolates = discriminator(interpolates, labels)

#     # Compute gradients of the discriminator w.r.t. the interpolated samples
#     gradients = autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=torch.ones_like(d_interpolates),
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]

#     gradients = gradients.view(gradients.size(0), -1)  # Flatten the gradients
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty


def smooth_labels(labels, smoothing):
    """ Apply label smoothing. """
    labels = labels.float()
    labels = labels * (1 - smoothing * 2) + smoothing
    return labels


def noisy_labels(labels, flip_prob):
    """ Apply label flipping with a certain probability. """
    if flip_prob > 0:
        flip_mask = torch.rand(labels.size()) < flip_prob
        labels[flip_mask] = 1 - labels[flip_mask]
    return labels


class PhonemeImageGAN(T2BGenInterface, nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        phoneme_cls: List[int],
        device: str,
        lr_g: float,
        lr_d: float,
        n_critic: int = 5,
        pool: str = "max",
        loss: str = "wasserstein",
    ):
        super(PhonemeImageGAN, self).__init__()
        self.starting = True

        assert isinstance(phoneme_cls, list)
        if len(phoneme_cls) == 1:
            self.conditional = False
        else:
            self.conditional = False
        
        self.phoneme_cls = phoneme_cls
        self.n_classes = len(self.phoneme_cls)
        self.input_shape = tuple(input_shape)

        if self.input_shape == (128, 8, 8):
            self._g = Generator_128_8_8(latent_dim, phoneme_cls).to(device)
            self._d = Discriminator_128_8_8(128, phoneme_cls).to(device)
        elif self.input_shape == (4, 64, 32):
            self._g = Generator_4_64_32(latent_dim, phoneme_cls).to(device)
            self._d = Discriminator_4_64_32(128, phoneme_cls, pool=pool).to(device)  # TODO
        else:
            raise ValueError(f"Invalid input shape: {input_shape}")
        
        print(f"self.conditional = {self.conditional}")
        print(f"self._g.conditional = {self._g.conditional}")
        print(f"self._d.conditional = {self._d.conditional}")
        
        print(f"self._g.__class__.__name__ = {self._g.__class__.__name__}")
        print(f"self._d.__class__.__name__ = {self._d.__class__.__name__}")
        
        self.device = device
        self.n_critic = n_critic
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.loss = loss
        print(f"self.loss = {self.loss}")

        self._g.apply(self._weights_init)
        self._d.apply(self._weights_init)

        self.init_optimizers()

    def init_optimizers(self) -> None:
        # self.optim_d = optim.AdamW(self._d.parameters(), lr=self.lr_d)
        # self.optim_g = optim.AdamW(self._g.parameters(), lr=self.lr_g)

        # self.optim_d = optim.SGD(self._d.parameters(), lr=self.lr_d, momentum=0.9)
        # self.optim_g = optim.Adam(self._g.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        
        # self.optim_d = optim.RMSprop(self._d.parameters(), lr=self.lr_d)
        # self.optim_g = optim.RMSprop(self._g.parameters(), lr=self.lr_g)

        self.optim_d = optim.Adam(self._d.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(self._g.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        

    def _weights_init(self, m):
        # if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=self.device).expand_as(real_samples)

        # interpolated samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        # discriminator's output for interpolated samples
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

        # compute the gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2))

        return gradient_penalty


    def train_step(self, dl):
        # update discriminator
        losses_D = []
        for _ in range(self.n_critic):
            X_real, y, _, _ = next(iter(dl))
            X_real = X_real.to(self.device)
            y = y.to(self.device)
            y = _get_indices_in_classes(y, torch.tensor(self.phoneme_cls, device=y.device)).to(y.device)

            self._d.zero_grad()

            output_real = self._d(X_real, y)
            noise = torch.randn(y.size(0), self._g.latent_dim, device=self.device)

            X_fake = self._g(noise, y)
            output_fake = self._d(X_fake.detach(), y)

            loss_D = -torch.mean(output_real) + torch.mean(output_fake)
            
            # compute gradient penalty
            gradient_penalty = self.compute_gradient_penalty(X_real, X_fake.detach(), y)
            
            # add gradient penalty to the discriminator loss
            lambda_gp = 10
            loss_D += lambda_gp * gradient_penalty

            loss_D.backward()

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
        y = _get_indices_in_classes(y, torch.tensor(self.phoneme_cls, device=y.device)).to(y.device)

        self._g.zero_grad()
        noise = torch.randn(y.size(0), self._g.latent_dim, device=self.device)
        X_fake = self._g(noise, y)
        output_fake = self._d(X_fake, y)
        loss_G = -torch.mean(output_fake)
        loss_G.backward()

        # grad_norms = []
        # for p in self._g.parameters():
        #     if p.grad is not None:
        #         grad_norms.append(p.grad.norm(2).item())
        # print(f"Generator Gradient Norms: {grad_norms}")

        self.optim_g.step()

        return loss_D, loss_G

    def generate(self, label: torch.Tensor):
        """Generate an image. Label is the class label, not the index."""
        noise = torch.randn(1, self._g.latent_dim, device=self.device)
        return self.generate_from_given_noise(noise=noise, label=label)

    def generate_from_given_noise(self, noise: torch.Tensor, label: torch.Tensor):
        """Generate an image. Label is the class label, not the index."""
        self._g.eval()
        assert noise.size(0) == 1
        assert noise.size(1) == self._g.latent_dim
        
        if self.conditional:
            y = _get_indices_in_classes(label, torch.tensor(self.phoneme_cls, device=label.device)).to(label.device)
            gen_img = self._g(noise, y)
        else:
            gen_img = self._g(noise)
        return gen_img


    def create_synthetic_phoneme_dataset(
        self, n_samples, neural_window_shape: tuple = (128, 8, 8)
    ):
        assert isinstance(neural_window_shape, tuple)

        classes = self.phoneme_cls
        if isinstance(classes, int):
            classes = [classes]

        label_distribution = [1.0 / len(classes)] * len(classes)

        neural_windows = []
        phoneme_labels = []

        for _ in tqdm(range(n_samples), desc="Creating synthetic phoneme dataset with {n_samples} samples"):
            label = np.random.choice(classes, p=label_distribution)
            label = torch.from_numpy(np.array([label])).to(self.device)

            neural_window = self.generate(label=label)
            neural_window = neural_window.view(neural_window.size(0), *neural_window_shape)

            neural_windows.append(neural_window.to("cpu"))
            phoneme_labels.append(label.to("cpu"))

        return SyntheticPhonemeDataset(neural_windows, phoneme_labels)

    def save_state_dict(self, path: str):
        print(f"Store model state dict to: {path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path):
        args = load_args(args_path)
        print(f"\nargs = {args}")

        phoneme_cls = args["phoneme_cls"]
        if "phoneme_cls" in args["phoneme_ds_filter"].keys():
            phoneme_cls = args["phoneme_ds_filter"]["phoneme_cls"]

        model = cls(
            latent_dim=args["latent_dim"],
            phoneme_cls=phoneme_cls,
            device=args["device"],
            n_critic=5 if "n_critic" not in args.keys() else args["n_critic"],
            lr=1e-4 if "lr" not in args.keys() else args["lr"],
        )
        model.load_state_dict(torch.load(weights_path))

        return model


class Generator_128_8_8(nn.Module):
    def __init__(self, latent_dim: int, phoneme_cls: List[int], dec_emb_dim: int = 32):
        super(Generator_128_8_8, self).__init__()

        self.input_shape = (128, 8, 8)
        self.phoneme_cls = phoneme_cls
        self.n_classes = len(phoneme_cls)

        self.dec_emb_dim = dec_emb_dim
        self.latent_dim = latent_dim
        input_dim = latent_dim

        self.conditional = False
        self.embedding = None
        
        if len(phoneme_cls) > 1:
            self.conditional = True
            self.embedding = nn.Embedding(len(phoneme_cls), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
 
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
    def __init__(self, latent_dim: int, phoneme_cls: List[int], dec_emb_dim: int = 32):
        super(Generator_4_64_32, self).__init__()

        self.input_shape = (4, 64, 32)
        self.phoneme_cls = phoneme_cls
        self.n_classes = len(phoneme_cls)

        self.dec_emb_dim = dec_emb_dim
        self.latent_dim = latent_dim
        input_dim = latent_dim

        self.conditional = False
        self.embedding = None
        
        if len(phoneme_cls) > 1:
            self.conditional = True
            self.embedding = nn.Embedding(len(phoneme_cls), self.dec_emb_dim)
            input_dim += self.dec_emb_dim
 
        print(f"input_dim = {input_dim}")
        dec_hidden_dim = 512
        self.dec_hidden_dim = dec_hidden_dim

        self.fc = nn.Sequential(nn.Linear(input_dim, dec_hidden_dim * 2 * 1), nn.LeakyReLU(0.2, inplace=True))
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


    def forward(self, noise: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass of the generator. Takes y as indices indices, not class labels. """
        
        if self.conditional:
            y_emb = self.embedding(y)
            h = torch.concat((noise, y_emb), dim=1)
            # y_one_hot = F.one_hot(y_indices, num_classes=len(self.classes)).float()
        else:
            h = noise

        h = self.fc(h)
        h = h.view(h.size(0), self.dec_hidden_dim, 2, 1)  # reshape to (dec_hidden_dim) x 2 x 1
        h = self.model(h)
        h = h.view(-1, 1, 256, 32)
        return h

class Discriminator_128_8_8(nn.Module):
    def __init__(self, n_channels: int, phoneme_cls: List[int]):
        super(Discriminator_128_8_8, self).__init__()
        self.phoneme_cls = phoneme_cls
        self.n_classes = len(phoneme_cls)
        self.input_shape = (128, 8, 8)

        if len(phoneme_cls) > 1:
            self.conditional = True
            input_dim = n_channels + self.n_classes
            self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        else:
            self.conditional = False
            input_dim = n_channels
            self.label_emb = None

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, 1, 1, bias=False),  # -> (64) x 8 x 8
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # -> (128) x 4 x 4
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (128) x 2 x 2
            nn.Conv2d(128, 256, 3, 1, 1),  # -> (256) x 2 x 2
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (256) x 1 x 1
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),  # -> (512)
            # nn.BatchNorm1d(512),
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
    def __init__(self, n_channels: int, phoneme_cls: List[int], pool: str = "max"):
        super(Discriminator_4_64_32, self).__init__()
        self.phoneme_cls = phoneme_cls
        self.n_classes = len(phoneme_cls)
        self.input_shape = (4, 64, 32)

        if len(phoneme_cls) > 1:
            self.conditional = True
            input_dim = n_channels + self.n_classes
            self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        else:
            self.conditional = False
            input_dim = n_channels
            self.label_emb = None

        if pool == "max":
            self.model = nn.Sequential(
                nn.Conv2d(4, 16, 3, 1, 1, bias=False),  # -> (16) x 64 x 32
                # nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # -> (16) x 32 x 16
                nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> (32) x 16 x 8
                # nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # -> (32) x 8 x 4
                nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> (64) x 4 x 2
                # nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 2 x 1
                # nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        elif pool == "avg":
            self.model = nn.Sequential(
                nn.Conv2d(4, 16, 3, 1, 1, bias=False),  # -> (16) x 64 x 32
                # nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, 2), # -> (16) x 32 x 16
                nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> (32) x 16 x 8
                # nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, 2), # -> (32) x 8 x 4
                nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> (64) x 4 x 2
                # nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 2 x 1
                # nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        elif pool == "none":
            self.model = nn.Sequential(
                nn.Conv2d(4, 16, 3, 1, 1, bias=False),  # -> (16) x 64 x 32
                # nn.InstanceNorm2d(16, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(16, 32, 3, 2, 1, bias=False),  # -> (32) x 32 x 16
                # nn.InstanceNorm2d(32, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # -> (64) x 16 x 8
                # nn.InstanceNorm2d(64, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # -> (128) x 8 x 4
                # nn.InstanceNorm2d(128, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # -> (64) x 4 x 2
                # nn.InstanceNorm2d(256, affine=True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 256, 3, 2, 1, bias=False),  # -> (256) x 2 x 1
                # nn.InstanceNorm2d(256, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.fc1 = nn.Linear(256 * 2 * 1, 512)
        self.fc2 = nn.Linear(512, 1)
        

    def forward(self, img: torch.Tensor, y: torch.Tensor):
        img = img.view(-1, *self.input_shape)
        if self.conditional:
            y = self.label_emb(y)
            y = y.view(y.size(0), y.size(1), 1, 1)
            y = y.repeat(1, 1, img.size(2), img.size(3))
            d_in = torch.cat((img, y), 1)
        else:
            d_in = img

        out = self.model(d_in)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

