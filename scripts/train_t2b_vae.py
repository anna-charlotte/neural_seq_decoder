import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from data.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.models.vae import VAE, ELBOLoss, GECOLoss
from text2brain.visualization import plot_brain_signal_animation
from utils import set_seeds


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
                SoftsignTransform(),
            ]
        )

    # load train and test data
    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": [3]},
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
        phoneme_ds_filter={"correctness_value": ["C"]},
        transform=transform,
    )

    # model = VAE(input_dim=256*32, hidden_dim=400, latent_dim=100).to(device)
    model = VAE(input_channels=1, latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    loss_fn = ELBOLoss(mse_reduction="sum")  # GECOLoss(goal=0.2, step_size=1e-2, mse_reduction='sum')
    loss_fn.to(device)

    # noise_vector = torch.randn(1, gan._g.latent_dim, device=gan.device)
    n_epochs = args["n_epochs"]

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    all_mse = []
    all_kld = []
    all_train_loss = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_dl):
            X, y, _, _ = data
            X = X.to(device)

            optimizer.zero_grad()
            X_recon, mu, logvar = model(X)

            mse, kld = loss_fn(X_recon, X, mu, logvar)
            loss = mse + kld
            # loss = loss_fn(X_recon, X, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # all_mse.append(mse.cpu().detach().numpy())
            # all_kld.append(kld.cpu().detach().numpy())
            all_train_loss.append(loss.cpu().detach().numpy())

            # output training stats
            if i % 500 == 0:
                print(f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] curr_loss: {loss.item()} ")
                plot_original_vs_reconstructed_image(
                    X[0][0].cpu().detach().numpy(),
                    X_recon[0][0].cpu().detach().numpy(),
                    ROOT_DIR / "evaluation" / "vae" / f"a_reconstructed_image_{epoch}_{i}.png",
                )

    plt.figure(figsize=(10, 6))
    plt.plot(all_mse, label="MSE", linewidth=0.5)
    plt.plot(all_kld, label="KLD", linewidth=0.5)
    plt.plot(all_train_loss, label="Training Loss", linewidth=0.5)

    # Adding titles and labels
    plt.title("Losses over Epochs")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()

    # Display the plot
    plt.savefig(ROOT_DIR / "evaluation" / "vae" / "losses_train.png")


def plot_original_vs_reconstructed_image(X: np.ndarray, X_recon: np.ndarray, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))

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
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    args["batch_size"] = 16

    args["input_dim"] = 100
    args["hidden_dim"] = 100
    args["latent_dim"] = 256

    args["n_epochs"] = 100
    args["lr"] = 0.0001

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
