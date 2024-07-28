import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
from text2brain.models.vae import VAE, ELBOLoss, GECOLoss, compute_mean_logvar_mse
from utils import set_seeds


def plot_means_and_stds(means: np.ndarray, stds: np.ndarray, phoneme: str):
    # FIRST PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.hist(means, bins=20, color="blue", edgecolor="black")
    ax1.set_title("Histogram of Mean Values (256 dim)")
    ax1.set_xlabel("Mean Value")
    ax1.set_ylabel("Frequency")

    ax2.hist(stds, bins=20, color="green", edgecolor="black")
    ax2.set_title("Histogram of Standard Deviation (STD) Values (256 dim))")
    ax2.set_xlabel("STD Value")
    ax2.set_ylabel("Frequency")

    fig.suptitle(f"VAE, Average Means and Standard Deviations for all phonemes (range(1, 39))", fontsize=16)
    plt.tight_layout()

    out_file = (
        ROOT_DIR
        / "evaluation"
        / "vae"
        / "latent_dim_evaluation"
        / f"average_means_and_stds__phoneme_{phoneme}__histogram.png"
    )
    plt.savefig(out_file)
    print(f"Saved plot to: {out_file}")

    # SECOND PLOT
    channels = list(range(1, len(means) + 1))  # channels from 1 to 256

    plt.figure(figsize=(12, 6))
    plt.scatter(channels, means, color="blue", label="Mean Values", marker="x")
    plt.scatter(channels, stds, color="green", label="STD Values", marker="o")

    plt.title("Mean and STD Values for Each Channel")
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.legend()

    out_file = (
        ROOT_DIR
        / "evaluation"
        / "vae"
        / "latent_dim_evaluation"
        / f"average_means_and_stds__phoneme_{phoneme}__per_channel.png"
    )
    plt.savefig(out_file)
    print(f"Saved plot to: {out_file}")


def main() -> None:
    # give model paths
    model_dir = Path(
        "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_20240724_230052"
    )
    weights_file = model_dir / "modelWeights_epoch_399"
    args_file = model_dir / "args"

    with open(args_file, "rb") as file:
        args = pickle.load(file)

    if "phoneme_cls" not in args.keys():
        args["phoneme_cls"] = list(range(1, 40))
    print(f"args = {args}")

    # compute average mean and logvar

    # sample from that

    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # load vae from args and state dict
    vae = VAE.load_model(args_file, weights_file)

    with open(args["train_set_path"], "rb") as handle:
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

    with open(ROOT_DIR / "evaluation" / "vae" / "latent_dim_evaluation" / "label2metrics.json", "r") as file:
        label2metrics = json.load(file)

    means = []
    stds = []
    n_samples = []

    for phoneme_dict in label2metrics.values():
        mean = np.array(phoneme_dict["mean"])
        std = np.array(phoneme_dict["logvar"])

        means.append(mean)
        stds.append(std)
        n_samples.append(phoneme_dict["n_samples"])

    weighted_means = np.average(means, axis=0, weights=n_samples)
    weighted_stds = np.average(stds, axis=0, weights=n_samples)

    print(f"weighted_means.shape = {weighted_means.shape}")
    print(f"weighted_stds.shape = {weighted_stds.shape}")
    plot_means_and_stds(means=weighted_means, stds=weighted_stds, phoneme="all")

    label2metrics = {}
    for phoneme in args["phoneme_cls"]:
        train_dl = get_data_loader(
            data=train_data,
            batch_size=1,
            shuffle=True,
            collate_fn=None,
            dataset_cls=PhonemeDataset,
            phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": [phoneme]},
            transform=transform,
        )
        n_samples = len(train_dl.dataset)

        label2metrics[phoneme] = {"mean": 0.0, "logvar": 0.0, "mse": 0.0, "n_samples": n_samples}

        results = compute_mean_logvar_mse(vae=vae, dl=train_dl)

        average_means = np.around(results.mean.detach().numpy(), decimals=3)
        average_stds = np.around(torch.exp(0.5 * results.logvar).detach().numpy(), decimals=3)

        label2metrics[phoneme]["mean"] = average_means.tolist()
        label2metrics[phoneme]["std"] = average_stds.tolist()
        label2metrics[phoneme]["mse"] = float(results.mse)

        plot_means_and_stds(
            means=average_means, stds=average_stds, phoneme=f"{phoneme} ({n_samples} samples)"
        )

        with open(
            ROOT_DIR / "evaluation" / "vae" / "latent_dim_evaluation" / "label2metrics.json", "w"
        ) as file:
            json.dump(label2metrics, file, indent=4)

        # for i, data in enumerate(train_dl):
        #     X, y, _, _ = data
        #     assert y.item() == phoneme
        #     X = X.to(device)

        #     X_recon, mu, logvar = vae(X)

        #     mse = F.mse_loss(X_recon, X, reduction='sum')
        #     label2metrics[phoneme]["mean"] += mu
        #     label2metrics[phoneme]["logvar"] += logvar
        #     label2metrics[phoneme]["mse"] += mse

        # assert i == len(train_dl.dataset) - 1, f"i = {i} != {len(train_dl.dataset)} = len(train_dl.dataset)"

        # label2metrics[phoneme]["mean"] += label2metrics[phoneme]["mean"] / label2metrics[phoneme]["n_samples"]
        # label2metrics[phoneme]["logvar"] += label2metrics[phoneme]["logvar"] / label2metrics[phoneme]["n_samples"]
        # label2metrics[phoneme]["mse"] += label2metrics[phoneme]["mse"] / label2metrics[phoneme]["n_samples"]


if __name__ == "__main__":
    main()
