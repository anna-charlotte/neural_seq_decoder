import json
import pickle
from datetime import datetime
from pathlib import Path

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
from text2brain.models.vae import VAE, ELBOLoss, GECOLoss
from text2brain.visualization import plot_brain_signal_animation
from utils import set_seeds


def plot_loss_mse_kld(all_losses, all_mse, all_kld, out_file) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # plot training loss on the first subplot
    axes[0].plot(all_losses, label="Training Loss (ELBOLoss)", linewidth=0.5)
    axes[0].set_title("Training Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()

    # plot MSE on the first subplot
    axes[1].plot(all_mse, label="MSE", linewidth=0.5)
    axes[1].set_title("MSE over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()

    # Plot KLD on the second subplot
    axes[2].plot(all_kld, label="KLD", linewidth=0.5)
    axes[2].set_title("KLD over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KLD Loss")
    axes[2].legend()

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
                GaussianSmoothing(256, kernel_size=20, sigma=2.0, dim=1),
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

    model = VAE(latent_dim=args["latent_dim"], input_shape=args["input_shape"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    if args["loss"] == "elbo":
        loss_fn = ELBOLoss(reduction="sum")  # "none"
    elif args["loss"] == "geco":
        loss_fn = GECOLoss(goal=0.2, step_size=1e-2, reduction="sum")
        loss_fn.to(device)
    print(f"loss_fn.__class__.__name__ = {loss_fn.__class__.__name__}")

    n_epochs = args["n_epochs"]

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    all_mse = []
    all_kld = []
    all_epoch_loss = []

    class_loss = {label: 0.0 for label in phoneme_cls}
    class_mse = {label: 0.0 for label in phoneme_cls}
    class_kld = {label: 0.0 for label in phoneme_cls}
    class_counts = {label: 0 for label in phoneme_cls}

    plot_dir = (
        ROOT_DIR
        / "evaluation"
        / "vae"
        / f"test_reconstructed_images_model_input_shape_{'_'.join(map(str, args['input_shape']))}__lr_{args['lr']}__loss_{args['loss']}__gaussiansmoothing_20_2.0__bs_{batch_size}__all_phoneme_classes_39"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = 0.0
        epoch_klds = 0.0
        epoch_mses = 0.0

        for i, data in enumerate(train_dl):
            X, y, _, _ = data
            X = X.to(device)

            optimizer.zero_grad()
            X_recon, mu, logvar = model(X)

            mse, kld = loss_fn(X_recon, X, mu, logvar)
            loss = mse + kld
            # loss = loss_fn(X_recon, X, mu, logvar)
            loss.backward()
            epoch_losses += loss.item()
            epoch_mses += mse.item()
            epoch_klds += kld.item()
            optimizer.step()

            # output training stats
            if i % 100 == 0:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_dl) + i)
                writer.add_scalar("MSE/Train", mse.item(), epoch * len(train_dl) + i)
                writer.add_scalar("KLD/Train", kld.item(), epoch * len(train_dl) + i)

            if i % 500 == 0:
                print(f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] curr_loss: {loss.item()} ")
                for j in range(10):
                    plot_original_vs_reconstructed_image(
                        X[j][0].cpu().detach().numpy(),
                        X_recon[j][0].cpu().detach().numpy(),
                        plot_dir / f"reconstructed_image_{epoch}_{i}__cls_{y[j]}.png",
                    )

                model.save_state_dict(out_dir / f"modelWeights_epoch_{epoch}")

        all_epoch_loss.append(epoch_loss / batch_size)
        all_mse.append(epoch_mse / batch_size)
        all_kld.append(epoch_kld / batch_size)

        writer.add_scalar("Epoch_Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Epoch_MSE/Train", epoch_mse, epoch)
        writer.add_scalar("Epoch_KLD/Train", epoch_kld, epoch)

        plot_loss_mse_kld(all_epoch_loss, all_mse, all_kld, out_file=plot_dir / "losses_train.png")

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
        for loss in ["elbo"]:  # ["elbo", "geco"]:
            for lr in [1e-3]:  # [1e-3, 1e-4, 1e-5]:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")

                args = {}
                args["seed"] = 0
                args["device"] = "cuda"
                args["batch_size"] = 64
                args["loss"] = loss
                args["phoneme_cls"] = list(range(1, 40))

                # args["input_dim"] = 100
                args["latent_dim"] = 256
                args["input_shape"] = input_shape

                args["n_epochs"] = 400
                args["lr"] = lr

                args["transform"] = "softsign"

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
