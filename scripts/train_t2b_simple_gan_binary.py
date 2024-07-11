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
    torch.use_deterministic_algorithms(True)

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    eval_dir = ROOT_DIR / "evaluation" / "gan_binary" / f"{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"eval_dir = {eval_dir}")

    device = args["device"]
    batch_size = args["batch_size"]
    phoneme_cls = args["phoneme_cls"]
    nc = args["nc"]
    nz = args["nz"]
    ngf = args["ngf"]
    ndf = args["ndf"]
    lr = args["lr"]
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
    netG = Generator(ngpu=1, nz=nz, ngf=ngf, nc=nc).to(device)
    netG.apply(weights_init)
    print(netG)
    
    # Create the Discriminator
    netD = Discriminator(ngpu=1, nc=nc, ndf=ndf).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    # create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(10, nz, 1, 1, device=device)

    # establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

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
            X, _, _, _ = data
            X = X.to(device)

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

                # compute distances and print
                
                compute_distances(
                    phoneme2avg_window,
                    {phoneme_cls[0]: s for s in fake},
                    distance_metric='cosine_sim',
                )

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



    # G_losses = []
    # D_losses = []
    # # noise_vector = torch.randn(1, gan._g.latent_dim, device=gan.device)
    # n_epochs = args["n_epochs"]

    # with open(out_dir / "args", "wb") as file:
    #     pickle.dump(args, file)
    # with open(out_dir / "args.json", "w") as file:
    #     json.dump(args, file, indent=4)

    # for epoch in range(n_epochs):
    #     print(f"epoch = {epoch}")
    #     for i, data in enumerate(train_dl):
    #         errD, errG = gan(data)

    #         # output training stats
    #         if i % 250 == 0:
    #             print(
    #                 f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {errD.item()} Loss_G: {errG.item()}"
    #             )

    #         #     phoneme = 2
    #         #     signal = gan._g(noise_vector, torch.tensor([phoneme]).to(device))
    #         #     plot_brain_signal_animation(
    #         #         signal=signal,
    #         #         save_path=ROOT_DIR
    #         #         / "plots"
    #         #         / "data_visualization"
    #         #         / f"phone_{phoneme}_FAKE_ep{epoch}_i_{i}.gif",
    #         #         title=f"Phoneme {phoneme}, Frame",
    #         #     )

    #         # if i == 0:
    #         #     X, _, _, _ = data

    #         #     phoneme = 2
    #         #     for j in range(X.size(0)):
    #         #         sub_signal = X[j, :, :]
    #         #         print(sub_signal.size())
    #         #         plot_brain_signal_animation(
    #         #             signal=sub_signal,
    #         #             save_path=ROOT_DIR
    #         #             / "plots"
    #         #             / "data_visualization"
    #         #             / f"phone_{phoneme}_REAL_sample_{j}.gif",
    #         #             title=f"Real sample, Phoneme {phoneme}, Frame",
    #         #         )

    #         # save losses for plotting
    #         G_losses.append(errG.item())
    #         D_losses.append(errD.item())

    #     # save GAN
    #     file = out_dir / f"modelWeights_epoch_{epoch}"
    #     print(f"Storing GAN weights to: {file}")
    #     gan.save_state_dict(file)
    #     # torch.save(gan.state_dict(), file)

    # plot_gan_losses(G_losses, D_losses, out_file=ROOT_DIR / "plots" / "data_visualization" / f"gan_losses_nclasses_{len(phoneme_cls)}.png")



if __name__ == "__main__":
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    
    args["n_classes"] = 2
    args["phoneme_cls"] = [3]  # 10: "DH", 22: "M"

    args["batch_size"] = 128
    args["n_channels"] = 128

    args["nc"] = 1  
    args["nz"] = 100  # args["latent_dim"] = 100
    args["ngf"] = 64
    args["ndf"] = 64

    args["num_epochs"] = 50
    # args["lr_generator"] = 0.0001
    # args["lr_discriminator"] = 0.0001
    args["lr"] = 0.0002
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
    args["output_dir"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_{timestamp}__nclasses_{args['n_classes']}"

    main(args)
