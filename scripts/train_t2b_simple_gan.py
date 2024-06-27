import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.neural_decoder_trainer import get_data_loader
from text2brain.visualization import plot_brain_signal_animation

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PhonemeImageGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, n_channels, ndf, ngf, device, n_critic=5, clip_value=0.01, lr=1e-4):
        super(PhonemeImageGAN, self).__init__()
        self.g = Generator(latent_dim, n_classes, ngf).to(device)
        self.d = Discriminator(n_channels, n_classes, ndf).to(device)
        self.device = device
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.lr = lr

        self.g.apply(self._weights_init)
        self.d.apply(self._weights_init)

        self.optim_d = optim.RMSprop(self.d.parameters(), lr=self.lr)
        self.optim_g = optim.RMSprop(self.g.parameters(), lr=self.lr)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, data):
        X_real, y, _, _ = data
        y = y.to(self.device)
        X_real = X_real.to(self.device)
        X_real = X_real.view(X_real.size(0), X_real.size(1), 16, 16)
        X_real = X_real.view(X_real.size(0), X_real.size(1), 16, 16)

        for _ in range(self.n_critic):
            self.d.zero_grad()
            output_real = self.d(X_real, y)
            noise = torch.randn(y.size(0), self.g.latent_dim, device=self.device)
            X_fake = self.g(noise, y)
            output_fake = self.d(X_fake.detach(), y)
            errD = -(torch.mean(output_real) - torch.mean(output_fake))
            errD.backward()
            self.optim_d.step()

            # Weight clipping
            for p in self.d.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

        # Update generator
        self.g.zero_grad()
        output_fake = self.d(X_fake, y)
        errG = -torch.mean(output_fake)
        errG.backward()
        self.optim_g.step()

        return errD, errG

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, ngf):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.ngf = ngf
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim + n_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, 32, 4, 2, 1, bias=False),  # Output 32 channels, stop upscaling here
            nn.Tanh()
            # Final state size. (32) x 16 x 16
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        gen_input = torch.cat((noise, labels), -1)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)
        output = self.model(gen_input)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_channels, n_classes, ndf):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            # input is (n_channels) x 16 x 16
            nn.Conv2d(n_channels + n_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 1 x 1
            nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
            # nn.Sigmoid()
            # output. 1 x 1 x 1
        )
    
    def forward(self, img, labels):
        labels = self.label_emb(labels)
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, img.size(2), img.size(3))
        d_in = torch.cat((img, labels), 1)
        output = self.model(d_in)
        return output.view(-1)


def main(args: dict) -> None:

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)
    
    device = args["device"]
    batch_size = args["batch_size"]
    n_classes = 41
    print(f"device = {device}")

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    train_dl = get_data_loader(
        data=train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=None, 
        dataset_cls=PhonemeDataset,
    )
    
    print(f"len(train_dl) = {len(train_dl)}")

    test_file = args["test_set_path"]
    with open(test_file, "rb") as handle:
        test_data = pickle.load(handle)

    test_dl = get_data_loader(
        data=test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=None, 
        dataset_cls=PhonemeDataset, 
    )
    print(f"len(test_dl) = {len(test_dl)}")

    n_channels = 32
    latent_dim = 100
    
    ngf = 64 
    ndf = 64 

    n_epochs = 50
    lr = 0.00005
    clip_value = 0.01
    n_critic = 15

    gan = PhonemeImageGAN(
        latent_dim=latent_dim, 
        n_classes=n_classes, 
        n_channels=n_channels, 
        ndf=ndf, 
        ngf=ngf, 
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )
    print(f"gan.g = {gan.g}")
    print(f"gan.d = {gan.d}")

    G_losses = []
    D_losses = []
    noise_vector = torch.randn(1, gan.g.latent_dim, device=gan.device)

    for epoch in range(n_epochs):
        for i, data in enumerate(train_dl):
            errD, errG = gan(data)

            # Output training stats
            if i % 250 == 0:
                print(f'[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

                phoneme = 2
                signal = gan.g(noise_vector, torch.tensor([phoneme]).to(device))
                plot_brain_signal_animation(
                    signal=signal, 
                    save_path=ROOT_DIR / "plots"/ "data_visualization" / f"phone_{phoneme}_ep{epoch}_i_{i}.gif" , 
                    title=f"Phoneme {phoneme}, Frame"
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    plot_gan_losses(G_losses, D_losses, out_file=ROOT_DIR / "plots" / "data_visualization" / "gan_losses.png")



def plot_gan_losses(g_losses: list, d_losses: list, out_file: Path) -> None:
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses,label="G")
    plt.plot(d_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_file)



if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 8
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    main(args)
