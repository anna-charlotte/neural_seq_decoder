import json
import pickle
from datetime import datetime
from pathlib import Path
import argparse
import os
from typing import Dict, List
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR, load_averaged_windows_for_all_classes, save_averaged_windows_for_all_classes
from neural_decoder.transforms import ReorderChannelTransform, SoftsignTransform, TransposeTransform, AddOneDimensionTransform
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.visualization import plot_brain_signal_animation
from utils import set_seeds


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# # Define a Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# # Generator with Residual Blocks
# class Generator(nn.Module):
#     def __init__(self, ngpu, nz, ngf, nc):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(8, 4), stride=(1, 1), padding=0, bias=False),  # (8, 4)
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             ResidualBlock(ngf * 8),
            
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),  # (16, 8)
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             ResidualBlock(ngf * 4),
            
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),  # (32, 16)
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             ResidualBlock(ngf * 2),
            
#             nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1), bias=False),  # (64, 16)
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             ResidualBlock(ngf),
            
#             nn.ConvTranspose2d(ngf, nc, kernel_size=(6, 4), stride=(4, 2), padding=1, bias=False),  # (256, 32)
#             nn.Tanh()
#         )

#     def forward(self, input):
#         return self.main(input)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=(8, 4), stride=(1, 1), padding=0, bias=False),  # (8, 4)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),  # (16, 8)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),  # (32, 16)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1), bias=False),  # (64, 16)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=(6, 4), stride=(4, 2), padding=1, bias=False),  # (256, 32)
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 32``
            nn.Conv2d(nc, ndf, kernel_size=(6, 4), stride=(4, 2), padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=(3, 3), stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=(8, 4), stride=(1, 1), padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



def compute_distances(
    averaged: Dict[int, torch.Tensor],
    samples: Dict[int, List[torch.Tensor]],
    distance_metric: str,
):
    for sample_phoneme_cls, samples_ in samples.items():
        for sample_tensor in samples_:
            distances = []
            for avg_phoneme_cls, avg_tensor in sorted(averaged.items()):
                if distance_metric == "frobenius":
                    distance = torch.norm(sample_tensor - avg_tensor, p="fro")
                elif distance_metric == "cosine_sim":
                    sample_tensor_flat = sample_tensor.reshape(-1)
                    avg_tensor_flat = avg_tensor.reshape(-1)
                    distance = 1 - torch.nn.functional.cosine_similarity(
                        sample_tensor_flat, avg_tensor_flat, dim=0
                    )
                elif distance_metric == "manhattan":
                    distance = torch.sum(torch.abs(sample_tensor - avg_tensor))
                elif distance_metric == "mse":
                    distance = torch.mean((sample_tensor - avg_tensor) ** 2)
                else:
                    raise ValueError(
                        f"Distance metric is invalid: {distance_metric}. Valid options are: ."
                    )
                distances.append(distance)

            # Rank the true class
            sorted_distances = sorted((dist, kls) for kls, dist in enumerate(distances))
            curr_rank = None
            for rank, (dist, kls) in enumerate(sorted_distances):
                if kls == sample_phoneme_cls:
                    curr_rank = rank
                    curr_dist = dist
                    break
            if curr_rank is None:
                raise ValueError("No rank found!")
        
            print(f"curr_rank = {curr_rank}")
            print(f"curr_dist = {curr_dist}")
        

class Generator64(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator64(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def plot_neural_window(window, out_file, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(
        window, cmap="plasma", aspect="auto"
    )  # , vmin=overall_min_value, vmax=overall_max_value)
    plt.colorbar()
    
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Channel")

    plt.savefig(out_file)
    plt.close()



def main(args: dict) -> None:

    set_seeds(args["seed"])
    # torch.use_deterministic_algorithms(True)

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    timestamp = args["timestamp"]
    eval_dir = ROOT_DIR / "evaluation" / "gan_binary" / f"{timestamp}__lrd_{args['lr_d']}__lrg_{args['lr_g']}__with_64x64_DCGAN"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"eval_dir = {eval_dir}")

    device = args["device"]
    batch_size = args["batch_size"]
    phoneme_cls = args["phoneme_cls"]
    nc = args["nc"]
    nz = args["nz"]
    ngf = args["ngf"]
    ndf = args["ndf"]
    lr_d = args["lr_d"]
    lr_g = args["lr_g"]
    beta1 = args["beta1"]
    num_epochs = args["num_epochs"]

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    transform = ReorderChannelTransform()
    if args["transform"] == "softsign":
        transform = transforms.Compose([
            TransposeTransform(), 
            ReorderChannelTransform(),
            AddOneDimensionTransform(dim=0),
            SoftsignTransform()
        ])

    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": phoneme_cls}
    if isinstance(phoneme_cls, int):
        phoneme_ds_filter["phoneme_cls"] = phoneme_cls
    args["phoneme_ds_filter"] = phoneme_ds_filter

    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )
    print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")

    # load or compute the average images
    avg_window_file = ROOT_DIR / "evaluation" / "phoneme_class_to_average_window_with_reordering.pt"
    phoneme2avg_window = load_averaged_windows_for_all_classes(avg_window_file)
    phoneme2avg_window = {p: phoneme2avg_window[p]["avg_window"] for p in phoneme2avg_window.keys()}

    # Create the generator
    # netG = Generator(ngpu=1, nz=nz, ngf=ngf, nc=nc).to(device)
    # netG = Generator(ngpu=1).to(device)
    netG = Generator64(ngpu=1, nz=nz, ngf=ngf, nc=nc).to(device)

    netG.apply(weights_init)
    
    # Create the Discriminator
    # netD = Discriminator(ngpu=1, nc=nc, ndf=ndf).to(device)
    # netD = Discriminator(ngpu=1).to(device)
    netD = Discriminator64(ngpu=1, nc=nc, ndf=ndf).to(device)
    netD.apply(weights_init)
    print(netD)
    print(netG)

    criterion = nn.BCELoss()

    # create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(10, nz, 1, 1, device=device)

    # establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    # training loop
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_dl, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            X_full, _, _, _ = data
            X_full = X_full.to(device)

            # Extract the first 128 rows
            X = X_full[:, :, :128, :]

            # Reshape to (64, 64)
            X = X.view(-1, 64, 64)
            X = X.unsqueeze(1).repeat(1, 3, 1, 1)

            netD.zero_grad()
            real_cpu = X.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_dl),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    window = fake[0][0]
                    plot_neural_window(
                        window, 
                        eval_dir / f"generated_image_epoch_{epoch}_batch_{i}.png",
                        f"Generated image (epoch: {epoch}, batch: {i})"
                    )
                    if i == 0:
                        for i in range(len(X)):
                            print(f"X_full.size() = {X_full.size()}")
                            plot_neural_window(
                                X_full[i][0].to("cpu"), 
                                eval_dir / f"0true_image_{i}_256_32.png",
                                f"True image"
                            )
                        for i in range(len(X)):
                            plot_neural_window(
                                X[0][0].to("cpu"), 
                                eval_dir / f"0true_image_{i}_64_64.png",
                                f"True image (cropped)"
                            )

                # compute distances and print
                
                # compute_distances(
                #     phoneme2avg_window,
                #     {phoneme_cls[0]: s for s in fake},
                #     distance_metric='cosine_sim',
                # )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(train_dl)-1)):
                
            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(eval_dir / f"gan_losses_binary.png")


if __name__ == "__main__":
    for lr_d, lr_g in zip([0.0002, 0.00001, 0.00001,  0.00001, 0.00002, 0.00002, 0.00005, 0.00005, 0.00005], [0.0002, 0.0001, 0.0002, 0.00005, 0.0001, 0.0002, 0.0001, 0.0002, 0.0005]):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        print("in main ...")
        args = {}
        args["seed"] = 0
        args["device"] = "cuda"
        
        args["n_classes"] = 2
        args["phoneme_cls"] = [31]  # 10: "DH", 22: "M"

        args["batch_size"] = 128
        args["n_channels"] = 128

        args["nc"] = 3 #1 
        args["nz"] = 100  # args["latent_dim"] = 100
        args["ngf"] = 64
        args["ndf"] = 64

        args["num_epochs"] = 150
        # args["lr_generator"] = 0.0001
        # args["lr_discriminator"] = 0.0001
        args["lr_g"] = lr_g  # 0.0002
        args["lr_d"] = lr_d  # 0.00002
        args["beta1"] = 0.5
        # args["clip_value"] = 0.01
        # args["n_critic"] = 1
        args["transform"] = "softsign"

        args["train_set_path"] = (
            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
        )
        args["test_set_path"] = (
            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
        )
        args["output_dir"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_{timestamp}__nclasses_{args['n_classes']}__with_64x64_DCGAN"
        args["timestamp"] = timestamp

        main(args)
