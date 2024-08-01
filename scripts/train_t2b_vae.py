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
from text2brain.models.vae import VAE, CondVAE, logvar_to_std
from text2brain.visualization import plot_brain_signal_animation, plot_means_and_stds
from utils import load_pkl, set_seeds


def plot_elbo_loss(
    all_losses: List[int], all_mse: List[int], all_kld: List[int], out_file: Path, title: str
) -> None:
    """
    Plot and save the training loss, mean squared error (MSE), and kullback leibler divergence (KLD) over epochs.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle(title, fontsize=16)

    # plot training loss on the first subplot
    axes[0].plot(all_losses, label="ELBOLoss", linewidth=0.5)
    axes[0].set_title("ELBOLoss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBOLoss")
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


def plot_geco_loss(
    all_losses: List[int],
    all_mse: List[int],
    all_kld: List[int],
    all_beta: List[int],
    out_file: Path,
    geco_goal: float,
    title: str,
) -> None:
    """
    Plot and save the training loss, mean squared error (MSE), kullback leibler divergence (KLD), and beta values over epochs, with a GECO goal line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # plot training loss on top left subplot
    axes[0, 0].plot(all_losses, label="GECOLoss", linewidth=0.5)
    axes[0, 0].set_title("GECOLoss over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("GECOLoss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # plot MSE on top right subplot
    axes[0, 1].plot(all_mse, label="MSE", linewidth=0.5)
    axes[0, 1].set_title("MSE over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE Loss")
    axes[0, 1].axhline(y=geco_goal, color="r", linestyle="--", linewidth=1)
    axes[0, 1].text(
        x=len(all_mse) - 1,
        y=geco_goal,
        s="GECO goal",
        color="r",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
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
    # with open(train_file, "rb") as handle:
    #     train_data = pickle.load(handle)
    train_data = load_pkl(train_file)

    transform = None
    if args["transform"] == "softsign":
        transform = transforms.Compose(
            [
                TransposeTransform(0, 1),
                ReorderChannelTransform(),
                AddOneDimensionTransform(dim=0),
                GaussianSmoothing(
                    256,
                    kernel_size=args["gaussian_smoothing_kernel_size"],
                    sigma=args["gaussian_smoothing_sigma"],
                    dim=1,
                ),
                SoftsignTransform(),
            ]
        )

    phoneme_cls = args["phoneme_cls"]
    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": phoneme_cls}

    # load train and test data
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

    val_file = args["val_set_path"]
    # with open(val_file, "rb") as handle:
    #     val_data = pickle.load(handle)
    val_data = load_pkl(val_file)

    val_dl = get_data_loader(
        data=val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )

    test_file = args["test_set_path"]
    # with open(test_file, "rb") as handle:
    #     test_data = pickle.load(handle)
    test_data = load_pkl(test_file)

    test_dl = get_data_loader(
        data=test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )

    # model = VAE(latent_dim=args["latent_dim"], input_shape=args["input_shape"], device=device)
    model = CondVAE(
        latent_dim=args["latent_dim"], input_shape=args["input_shape"], classes=phoneme_cls, device=device, dec_emb_dim=args["dec_emb_dim"]
    )
    args["model_class"] = model.__class__.__name__

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    if args["loss"] == "elbo":
        loss_fn = ELBOLoss(reduction=args["loss_reduction"])
    elif args["loss"] == "geco":
        loss_fn = GECOLoss(
            goal=args["geco_goal"],
            step_size=args["geco_step_size"],
            beta_init=args["geco_beta_init"],
            beta_max=args["geco_beta_max"],
            reduction=args["loss_reduction"],
            device=device,
        )

    print(f"loss_fn.__class__.__name__ = {loss_fn.__class__.__name__}")

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    all_mse = {"train": [], "val": []}
    all_kld = {"train": [], "val": []}
    all_epoch_loss = {"train": [], "val": []}
    all_betas = []

    plot_dir = (
        ROOT_DIR
        / "evaluation"
        / "vae_conditional"
        / "run_20240801_1_phoneme_3_31"
        / f"reconstructed_images_model_input_shape_{'_'.join(map(str, args['input_shape']))}__dec_emb_dim_{args['dec_emb_dim']}__latent_dim_{args['latent_dim']}__loss_{args['loss']}_{args['geco_goal']}_{args['geco_step_size']}_{args['geco_beta_init']}_{args['geco_beta_max']}__lr_{args['lr']}__gs_{args['gaussian_smoothing_kernel_size']}_{args['gaussian_smoothing_sigma']}__bs_{batch_size}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot_dir = {plot_dir}")

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)
    with open(plot_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    print("\nArguments:")
    for k, v in args.items():
        print(f"\t{k} = {v}")

    n_batches_train = len(train_dl)
    n_batches_val = len(val_dl)

    n_epochs = args["n_epochs"]

    for epoch in range(n_epochs):
        epoch_train_loss, epoch_train_mse, epoch_train_kld = 0.0, 0.0, 0.0
        all_mu = []
        all_std = []

        model.train()
        for i, data in enumerate(train_dl):
            X, y, _, _ = map(lambda x: x.to(device), data)

            optimizer.zero_grad()
            X_recon, mu, logvar = model(X, y)
            if epoch % 10 == 0:
                all_mu.append(mu.detach().cpu().numpy())
                std = logvar_to_std(logvar.detach().cpu())
                all_std.append(std.numpy())

            results = loss_fn(X_recon, X, mu, logvar)
            results.loss.backward()
            optimizer.step()

            epoch_train_loss += results.loss.item()
            epoch_train_mse += results.mse.item()
            epoch_train_kld += results.kld.item()

            if isinstance(loss_fn, GECOLoss):
                all_betas.append(loss_fn.beta.cpu().detach().numpy().tolist())

            # output training stats
            if i % 100 == 0:
                writer.add_scalar("Loss/Train", results.loss.item(), epoch * len(train_dl) + i)
                writer.add_scalar("MSE/Train", results.mse.item(), epoch * len(train_dl) + i)
                writer.add_scalar("KLD/Train", results.kld.item(), epoch * len(train_dl) + i)

            if i % 500 == 0 and epoch % 10 == 0:
                for j in range(10):
                    plot_original_vs_reconstructed_image(
                        X[j][0].cpu().detach().numpy(),
                        X_recon[j][0].cpu().detach().numpy(),
                        plot_dir / f"reconstructed_image_{epoch}_{i}__cls_{y[j]}.png",
                    )

        all_epoch_loss["train"].append(epoch_train_loss / n_batches_train)
        all_mse["train"].append(epoch_train_mse / n_batches_train)
        all_kld["train"].append(epoch_train_kld / n_batches_train)

        # compute the mean and logvar
        if epoch % 10 == 0:
            mean_mu = np.mean(np.concatenate(all_mu, axis=0), axis=0)
            mean_std = np.mean(np.concatenate(all_std, axis=0), axis=0)
            print(f"mean_mu.shape = {mean_mu.shape}")
            print(f"mean_std.shape = {mean_std.shape}")
            assert mean_mu.shape == (args["latent_dim"],)
            assert mean_std.shape == (args["latent_dim"],)

            plot_means_and_stds(means=mean_mu, stds=mean_std, phoneme=f"3_31__epoch_{epoch}", out_dir=plot_dir)

        # validate on val set at end of epoch
        epoch_val_loss, epoch_val_mse, epoch_val_kld = 0.0, 0.0, 0.0

        model.eval()
        for i, data in enumerate(val_dl):
            X, y, _, _ = map(lambda x: x.to(device), data)

            X_recon, mu, logvar = model(X, y)
            results = loss_fn(X_recon, X, mu, logvar)

            epoch_val_loss += results.loss.item()
            epoch_val_mse += results.mse.item()
            epoch_val_kld += results.kld.item()

        all_epoch_loss["val"].append(epoch_val_loss / n_batches_val)
        all_mse["val"].append(epoch_val_mse / n_batches_val)
        all_kld["val"].append(epoch_val_kld / n_batches_val)

        writer.add_scalar("Epoch_Loss/Val", epoch_val_loss / n_batches_val, epoch)
        writer.add_scalar("Epoch_MSE/Val", epoch_val_mse / n_batches_val, epoch)
        writer.add_scalar("Epoch_KLD/Val", epoch_val_kld / n_batches_val, epoch)

        print(
            f"[{epoch}/{n_epochs}] train_loss: {epoch_train_loss / n_batches_train} val_loss: {epoch_val_loss / n_batches_val}"
        )
        model.save_state_dict(out_dir / f"modelWeights")

        if isinstance(loss_fn, ELBOLoss):
            plot_elbo_loss(
                all_epoch_loss["train"],
                all_mse["train"],
                all_kld["train"],
                out_file=plot_dir / "losses_train.png",
                title="Training Metrics",
            )
            plot_elbo_loss(
                all_epoch_loss["val"],
                all_mse["val"],
                all_kld["val"],
                out_file=plot_dir / "losses_val.png",
                title="Validation Metrics",
            )
        elif isinstance(loss_fn, GECOLoss):
            plot_geco_loss(
                all_epoch_loss["train"],
                all_mse["train"],
                all_kld["train"],
                all_betas,
                out_file=plot_dir / "losses_train.png",
                geco_goal=loss_fn.goal,
                title="Training Metrics",
            )
            plot_geco_loss(
                all_epoch_loss["val"],
                all_mse["val"],
                all_kld["val"],
                all_betas,
                out_file=plot_dir / "losses_val.png",
                geco_goal=loss_fn.goal,
                title="Validation Metrics",
            )

        # save training statistics
        with open(plot_dir / "trainingStats.json", "w") as file:
            json.dump(
                {"loss": all_epoch_loss, "mse": all_mse, "kld": all_kld, "beta": all_betas}, file, indent=4
            )

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
    print("In training ...")

    for input_shape in [(128, 8, 8)]:  # [(4, 64, 32), (1, 256, 32), (128, 8, 8)]:
        for loss in ["geco"]:  # ["elbo", "geco"]:
            for lr in [1e-4]:  # [1e-3, 1e-4, 1e-5]:
                for latent_dim in [256]:  # , 128, 100]:
                    for geco_goal in [0.045]:  # , 0.037]:
                        for geco_step_size in [1e-2]:  # , 1e-3, 1e-4]:
                            for geco_beta_init in [1e-3]:  # , 1e-5, 1e-7]:
                                for dec_emb_dim in [32]:
                                    for geco_beta_max in [1e-3, 2e-3, 3e-3, 4e-4, 1e-4]:

                                        now = datetime.now()
                                        timestamp = now.strftime("%Y%m%d_%H%M%S")

                                        args = {}
                                        args["seed"] = 0
                                        args["device"] = "cuda"
                                        args["batch_size"] = 64
                                        args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                                        args["dec_emb_dim"] = dec_emb_dim

                                        args["loss"] = loss
                                        args["loss_reduction"] = "mean"
                                        if loss == "geco":
                                            args["geco_goal"] = geco_goal
                                            args["geco_beta_init"] = geco_beta_init
                                            args["geco_step_size"] = geco_step_size
                                            args["geco_beta_max"] = geco_beta_max

                                        # args["input_dim"] = 100
                                        args["latent_dim"] = latent_dim
                                        args["input_shape"] = input_shape

                                        args["n_epochs"] = 100
                                        args["lr"] = lr

                                        args["transform"] = "softsign"
                                        args["gaussian_smoothing_kernel_size"] = 20
                                        args["gaussian_smoothing_sigma"] = 2.0

                                        args["train_set_path"] = (
                                            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
                                        )
                                        args["val_set_path"] = (
                                            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl"
                                        )
                                        args["test_set_path"] = (
                                            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl"
                                        )
                                        args["output_dir"] = (
                                            f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_{timestamp}"
                                        )

                                        main(args)
