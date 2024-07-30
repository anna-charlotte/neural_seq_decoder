import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data.augmentations import GaussianSmoothing
from data.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from text2brain.models.loss import ELBOLoss, GECOLoss
from text2brain.models.vae import VAE, CondVAE
from text2brain.visualization import plot_brain_signal_animation
from utils import set_seeds


def plot_elbo_loss(all_losses: List[int], all_mse: List[int], all_kld: List[int], out_file: Path) -> None:
    """
    Plot and save the training loss, mean squared error (MSE), and kullback leibler divergence (KLD) over epochs.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # plot training loss on the first subplot
    axes[0].plot(all_losses, label="Training Loss (ELBOLoss)", linewidth=0.5)   
    axes[0].set_title("Training Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()

    # plot MSE on the second subplot
    axes[1].plot(all_mse, label="MSE", linewidth=0.5)
    axes[1].set_title("MSE over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()

    # plot KLD on the third subplot
    axes[2].plot(all_kld, label="KLD", linewidth=0.5)
    axes[2].set_title("KLD over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KLD Loss")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_geco_loss(all_losses: List[int], all_mse: List[int], all_kld: List[int], all_beta: List[int], out_file: Path, geco_goal: float) -> None:
    """
    Plot and save the training loss, mean squared error (MSE), kullback leibler divergence (KLD), and beta values over epochs, with a GECO goal line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # plot training loss on top left subplot
    axes[0, 0].plot(all_losses, label="Training Loss (GECOLoss)", linewidth=0.5)
    axes[0, 0].set_title("Training Loss over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # plot MSE on top right subplot
    axes[0, 1].plot(all_mse, label="MSE", linewidth=0.5)
    axes[0, 1].set_title("MSE over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE Loss")
    axes[0, 1].axhline(y=geco_goal, color='r', linestyle='--', linewidth=1)
    axes[0, 1].text(x=len(all_mse) - 1, y=geco_goal, s="GECO goal", color='r', verticalalignment='bottom', horizontalalignment='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # plot KLD on lower left subplot
    axes[1, 0].plot(all_kld, label="KLD", linewidth=0.5)
    axes[1, 0].set_title("KLD over Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KLD Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # plot betas per n batch, while the otheras are plotted per epoch
    axes[1, 1].plot(all_beta, label="Beta", linewidth=0.5)
    axes[1, 1].set_title("Beta over Epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Beta Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main(args: dict) -> None:
    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args["device"]
    batch_size = args["batch_size"]

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    transform = None
    if args["transform"] == "softsign":
        transform = transforms.Compose(
            [
                TransposeTransform(0, 1),
                ReorderChannelTransform(),
                AddOneDimensionTransform(dim=0),
                GaussianSmoothing(256, kernel_size=args["gaussian_smoothing_kernel_size"], sigma=args["gaussian_smoothing_sigma"], dim=1),
                SoftsignTransform(),
            ]
        )

    phoneme_cls = args["phoneme_cls"]

    # load train and test data
    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": phoneme_cls},
        transform=transform,
    )

    print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")

    test_file = args["test_set_path"]
    with open(test_file, "rb") as handle:
        test_data = pickle.load(handle)

    test_dl = get_data_loader(
        data=test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": phoneme_cls},
        transform=transform,
    )

    # model = VAE(latent_dim=args["latent_dim"], input_shape=args["input_shape"], device=device)
    model = CondVAE(
        latent_dim=args["latent_dim"], input_shape=args["input_shape"], classes=phoneme_cls, device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    if args["loss"] == "elbo":
        loss_fn = ELBOLoss(reduction=args["loss_reduction"])
    elif args["loss"] == "geco":
        loss_fn = GECOLoss(
            goal=args["geco_goal"], 
            step_size=args["geco_step_size"],
            beta_init=args["geco_beta_init"], 
            reduction=args["loss_reduction"], 
            device=device
        )

    print(f"loss_fn.__class__.__name__ = {loss_fn.__class__.__name__}")

    n_epochs = args["n_epochs"]

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    all_mse = []
    all_kld = []
    all_epoch_loss = []
    all_betas = []

    plot_dir = (
        ROOT_DIR
        / "evaluation"
        / "vae_conditional"
        / "run_20240730_phoneme_3_31"
        / f"reconstructed_images_model_input_shape_{'_'.join(map(str, args['input_shape']))}__loss_{args['loss']}_{args['geco_goal']}_{args['geco_step_size']}__lr_{args['lr']}__gs_{args['gaussian_smoothing_kernel_size']}_{args['gaussian_smoothing_sigma']}__bs_{batch_size}__latent_dim_{args['latent_dim']}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)


    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)
    with open(plot_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    n_batches = len(train_dl)

    print("\nArguments:")
    for k, v in args.items():
        print(f"\t{k} = {v}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kld = 0.0
        epoch_mse = 0.0

        for i, data in enumerate(train_dl):
            X, y, _, _ = data
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            X_recon, mu, logvar = model(X, y)

            results = loss_fn(X_recon, X, mu, logvar)
            loss = results.loss
            mse = results.mse
            kld = results.kld
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_kld += kld.item()
            if isinstance(loss_fn, GECOLoss):
                all_betas.append(loss_fn.beta.cpu().detach().numpy())

            # output training stats
            if i % 100 == 0:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_dl) + i)
                writer.add_scalar("MSE/Train", mse.item(), epoch * len(train_dl) + i)
                writer.add_scalar("KLD/Train", kld.item(), epoch * len(train_dl) + i)
            
            if i % 1000 == 0:
                print(f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] curr_loss: {loss.item()} ")
                model.save_state_dict(out_dir / f"modelWeights")
                
            if i % 500 == 0 and epoch % 10 == 0:
                for j in range(10):
                    plot_original_vs_reconstructed_image(
                        X[j][0].cpu().detach().numpy(),
                        X_recon[j][0].cpu().detach().numpy(),
                        plot_dir / f"reconstructed_image_{epoch}_{i}__cls_{y[j]}.png",
                    )


        all_epoch_loss.append(epoch_loss / n_batches)
        all_mse.append(epoch_mse / n_batches)
        all_kld.append(epoch_kld / n_batches)

        writer.add_scalar("Epoch_Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Epoch_MSE/Train", epoch_mse, epoch)
        writer.add_scalar("Epoch_KLD/Train", epoch_kld, epoch)

        if isinstance(loss_fn, ELBOLoss):
            plot_elbo_loss(all_epoch_loss, all_mse, all_kld, out_file=plot_dir / "losses_train.png")
        elif isinstance(loss_fn, GECOLoss):
            plot_geco_loss(all_epoch_loss, all_mse, all_kld, all_betas, out_file=plot_dir / "losses_train.png", geco_goal=loss_fn.goal)

    writer.close()


def plot_original_vs_reconstructed_image(X: np.ndarray, X_recon: np.ndarray, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))

    # Set the color range
    vmin, vmax = -1, 1

    # Display the first image
    im0 = axes[0].imshow(X, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[0].axis("off")
    axes[0].set_title("Original image")

    # Display the second image
    im1 = axes[1].imshow(X_recon, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].axis("off")
    axes[1].set_title("Reconstructed image")

    # Add colorbars
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])

    # Display the plot
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":

    for input_shape in [(128, 8, 8)]:  # [(4, 64, 32), (1, 256, 32), (128, 8, 8)]:
        for loss in ["geco"]:  # ["elbo", "geco"]:
            for lr in [1e-5, 1e-4, 1e-3]:  # [1e-3, 1e-4, 1e-5]:
                for latent_dim in [256, 128, 100]:
                    for geco_step_size in [1e-2]:
                        for geco_goal in [0.02]:

                            if not (
                                (latent_dim==256 and lr==1e-4)
                                or lr==1e-5
                            ):
                                now = datetime.now()
                                timestamp = now.strftime("%Y%m%d_%H%M%S")

                                args = {}
                                args["seed"] = 0
                                args["device"] = "cuda"
                                args["batch_size"] = 64
                                args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                                
                                args["loss"] = loss
                                args["loss_reduction"] = "mean"
                                if loss == "geco":
                                    args["geco_goal"] = geco_goal
                                    args["geco_beta_init"] = 0.0000001
                                    args["geco_step_size"] = geco_step_size

                                # args["input_dim"] = 100
                                args["latent_dim"] = latent_dim
                                args["input_shape"] = input_shape

                                args["n_epochs"] = 80
                                args["lr"] = lr

                                args["transform"] = "softsign"
                                args["gaussian_smoothing_kernel_size"] = 20
                                args["gaussian_smoothing_sigma"] = 2.0

                                args["train_set_path"] = (
                                    "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
                                )
                                args["test_set_path"] = (
                                    "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
                                )
                                args["output_dir"] = (
                                    f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_{timestamp}"
                                )

                                main(args)
