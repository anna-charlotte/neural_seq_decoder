import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from neural_decoder.dataset import SyntheticPhonemeDataset


def phonemes_to_signal(model, phonemes: list) -> torch.Tensor:
    assert model.conditional == True

    unique_phonemes = list(set(phonemes))
    assert all(p in model.phoneme_cls for p in unique_phonemes)

    signal = []
    for p in phonemes:
        s = model.generate(
            label=torch.tensor(
                [
                    p,
                ]
            )
        )
        s = s.view(1, 32, 16, 16)
        signal.append(s)

    return torch.cat(signal, dim=1)


class PhonemeImageGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        phoneme_cls: Optional[list],
        n_channels: int,
        ndf: int,
        ngf: int,
        device: str,
        n_critic: int = 5,
        clip_value: float = 0.01,
        lr=1e-4,
    ):
        super(PhonemeImageGAN, self).__init__()
        if isinstance(phoneme_cls, list):
            self.conditional = True
        else:
            self.conditional = False

        self.phoneme_cls = phoneme_cls
        if isinstance(self.phoneme_cls, int):
            self.n_classes = 1
        elif isinstance(self.phoneme_cls, list):
            self.n_classes = len(self.phoneme_cls)

        self._g = Generator(latent_dim, phoneme_cls, ngf).to(device)
        self._d = Discriminator(n_channels, phoneme_cls, ndf).to(device)
        self.device = device
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.lr = lr

        self._g.apply(self._weights_init)
        self._d.apply(self._weights_init)

        self.optim_d = optim.RMSprop(self._d.parameters(), lr=self.lr)
        self.optim_g = optim.RMSprop(self._g.parameters(), lr=self.lr)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, data):
        X_real, y, _, _ = data

        y = y.to(self.device)
        y = _get_indices_in_classes(y, torch.tensor(self.phoneme_cls, device=y.device)).to(y.device)
        X_real = X_real.to(self.device)
        # X_real = X_real.view(X_real.size(0), X_real.size(1), 16, 16)
        X_real = X_real.view(-1, 128, 8, 8)
        # if isinstance(self.phoneme_cls, list) and len(self.phoneme_cls) < 41:
        #     # y = y - 1
        #     y = _get_indices_in_classes(y, torch.tensor(self.phoneme_cls, device=y.device)).to(y.device)

        for _ in range(self.n_critic):
            self._d.zero_grad()

            output_real = self._d(X_real, y)
            noise = torch.randn(y.size(0), self._g.latent_dim, device=self.device)

            X_fake = self._g(noise, y)
            output_fake = self._d(X_fake.detach(), y)
            errD = -(torch.mean(output_real) - torch.mean(output_fake))
            errD.backward()
            self.optim_d.step()

            # Weight clipping
            for p in self._d.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

        # Update generator
        self._g.zero_grad()
        output_fake = self._d(X_fake, y)
        errG = -torch.mean(output_fake)
        errG.backward()
        self.optim_g.step()

        return errD, errG

    def generate(self, label: torch.Tensor):
        self._g.eval()
        noise = torch.randn(1, self._g.latent_dim, device=self.device)
        gen_img = self._g(noise, label.to(self.device))
        return gen_img

    def create_synthetic_phoneme_dataset(
        self, n_samples, label_distribution: list = None, neural_window_shape: tuple = (128, 8, 8)
    ):
        assert isinstance(neural_window_shape, tuple)

        classes = self.phoneme_cls
        if isinstance(classes, int):
            classes = [classes]

        if label_distribution is None:
            label_distribution = [1.0 / len(classes)] * len(classes)

        assert len(label_distribution) == len(classes), "Label distribution must match the number of classes"
        assert np.isclose(sum(label_distribution), 1.0), "Label distribution must sum to 1"

        neural_windows = []
        phoneme_labels = []

        for _ in tqdm(range(n_samples), desc="Creating synthetic phoneme dataset with {n_samples} samples"):
            label = np.random.choice(classes, p=label_distribution)
            label = torch.from_numpy(np.array([label])).to(self.device)
            y = _get_indices_in_classes(label, torch.tensor(classes, device=label.device)).to(label.device)

            neural_window = self.generate(label=y)
            neural_window = neural_window.view(neural_window.size(0), *neural_window_shape)

            neural_windows.append(neural_window.to("cpu"))
            phoneme_labels.append(label.to("cpu"))

        return SyntheticPhonemeDataset(neural_windows, phoneme_labels)

    def save_state_dict(self, path: str):
        print(f"Store model state dict to: {path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, args_path: Path, weights_path: Path):
        with open(args_path, "rb") as file:
            args = pickle.load(file)

        print(f"\nargs = {args}")
        phoneme_cls = args["phoneme_cls"]
        if "phoneme_cls" in args["phoneme_ds_filter"].keys():
            phoneme_cls = args["phoneme_ds_filter"]["phoneme_cls"]

        model = cls(
            latent_dim=args["latent_dim"],
            phoneme_cls=phoneme_cls,
            n_channels=args["n_channels"],
            ndf=args["ndf"],
            ngf=args["ngf"],
            device=args["device"],
            n_critic=5 if "n_critic" not in args.keys() else args["n_critic"],
            clip_value=0.01 if "clip_value" not in args.keys() else args["clip_value"],
            lr=1e-4 if "clip_value" not in args.keys() else args["clip_value"],
        )
        model.load_state_dict(torch.load(weights_path))
        # with open(weights_path, "rb") as file:
        # model.load_state_dict(torch.load(file))

        return model


class Generator(nn.Module):
    def __init__(self, latent_dim, phoneme_cls, ngf):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.phoneme_cls = phoneme_cls
        self.ngf = ngf
        if isinstance(phoneme_cls, list):
            self.conditional = True
            n_classes = len(phoneme_cls)
            input_dim = latent_dim + n_classes
            self.label_emb = nn.Embedding(n_classes, n_classes)
        else:
            self.conditional = False
            input_dim = latent_dim
            self.label_emb = None

        # self.model = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, 32, 4, 2, 1, bias=False),  # Output 32 channels, stop upscaling here
        #     nn.Tanh(),
        #     # Final state size. (32) x 16 x 16
        # )

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, 128, 3, 1, 1, bias=False),  # Output 128 channels, stop upscaling here
            nn.Tanh(),
            # Fina state size. (128) x 8 x 8
        )

    def forward(self, noise, labels):

        if self.conditional:
            labels = self.label_emb(labels)
            gen_input = torch.cat((noise, labels), -1)
            gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)
        else:
            gen_input = noise.view(noise.size(0), -1, 1, 1)
        output = self.model(gen_input)
        return output


def _get_indices_in_classes(labels, classes):
    indices = []
    for label in labels:
        index = torch.where(classes == label)[0]
        indices.append(index)
        if not index.numel() > 0:
            raise ValueError(f"Invalid label given: {label}")

    return torch.tensor(indices).int()


class Discriminator(nn.Module):
    def __init__(self, n_channels, phoneme_cls, ndf):
        super(Discriminator, self).__init__()

        if isinstance(phoneme_cls, list):
            self.conditional = True
            n_classes = len(phoneme_cls)
            input_dim = n_channels + n_classes
            self.label_emb = nn.Embedding(n_classes, n_classes)
        else:
            self.conditional = False
            input_dim = n_channels
            self.label_emb = None

        self.phoneme_cls = phoneme_cls
        print(f"phoneme_cls = {phoneme_cls}")
        # self.model = nn.Sequential(
        #     # input is (n_channels) x 16 x 16
        #     nn.Conv2d(input_dim, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (ndf) x 8 x 8
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (ndf*2) x 4 x 4
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
        #     # nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size: (ndf*4) x 1 x 1
        #     nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
        #     # nn.Sigmoid()
        #     # output. 1 x 1 x 1
        # )

        self.model = nn.Sequential(
            # input is (input_dim) x 8 x 8
            nn.Conv2d(input_dim, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 2 x 2
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 1 x 1
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            # output. 1 x 1 x 1
        )

    def forward(self, img, labels):

        if self.conditional:
            labels = self.label_emb(labels)
            labels = labels.view(labels.size(0), labels.size(1), 1, 1)
            labels = labels.repeat(1, 1, img.size(2), img.size(3))
            d_in = torch.cat((img, labels), 1)
        else:
            d_in = img
        output = self.model(d_in)
        return output.view(-1)
