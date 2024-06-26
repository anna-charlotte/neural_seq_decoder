import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.neural_decoder_trainer import get_data_loader

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


class Generator(nn.Module):
    def __init__(self, latent_dim, ngf):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
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

    def forward(self, input):
        output = self.model(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_channels, ndf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (n_channels) x 16 x 16
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
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

    def forward(self, input):
        output = self.model(input)
        return output.view(-1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def main(args: dict) -> None:

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)
    
    device = args["device"]
    batch_size = args["batch_size"]
    phoneme_cls = 2
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
        phoneme_cls=phoneme_cls,
    )
    phonemes = []
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
        phoneme_cls=phoneme_cls,
    )
    print(f"len(test_dl) = {len(test_dl)}")

    n_channels = 32
    latent_dim = 100
    
    ngf = 64  # Size of feature maps in the generator
    ndf = 64  # Size of feature maps in discriminator

    n_epochs = 50
    lr = 0.00005
    clip_value = 0.01  # Weight clipping value
    n_critic = 3  # Number of critic iterations per generator iteration


    gen = Generator(latent_dim, ngf).to(device)
    dis = Discriminator(n_channels, ndf).to(device)

    gen.apply(weights_init)
    dis.apply(weights_init)

    optimizerD = optim.RMSprop(dis.parameters(), lr=lr)
    optimizerG = optim.RMSprop(gen.parameters(), lr=lr)

    G_losses = []
    D_losses = []

    for epoch in range(n_epochs):
        for i, data in enumerate(train_dl):
            X_real, y, logits, dayIdx = data
            X_real = X_real.to(device)
            X_real = X_real.view(X_real.size(0), X_real.size(1), 16, 16)

            # Update discriminator n_critic times
            for _ in range(n_critic):
                dis.zero_grad()
                output_real = dis(X_real)
                noise = torch.randn(y.size(0), latent_dim, 1, 1, device=device)
                X_fake = gen(noise)
                output_fake = dis(X_fake.detach())
                errD = -(torch.mean(output_real) - torch.mean(output_fake))
                errD.backward()
                optimizerD.step()

                # Weight clipping
                for p in dis.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Update generator
            gen.zero_grad()
            output_fake = dis(X_fake)
            errG = -torch.mean(output_fake)
            errG.backward()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                if i % 50 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         fake = gen(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # iters += 1
    
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
