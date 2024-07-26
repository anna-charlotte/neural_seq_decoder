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


def main(args: dict) -> None:
    # give model paths
    model_dir = Path(
        "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_20240724_230052"
    )
    weights_file = model_dir / "modelWeights_epoch_399"
    args_file = model_dir / "args"

    with open(args_file, "rb") as file:
        args = pickle.load(file)

    # compute average mean and logvar

    # sample from that

    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args["device"]

    # load vae from args and state dict
    vae = VAE.load_model(model_dir / "args", model_dir / "modelWeights")
    print(f"vae = {vae}")

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

    label2metrics = {}
    for phoneme in args["phoneme_cls"]:
        train_dl = get_data_loader(
            data=train_data,
            batch_size=1,
            shuffle=True,
            collate_fn=None,
            dataset_cls=PhonemeDataset,
            phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": phoneme},
            transform=transform,
        )
        print(f"phoneme = {phoneme}")
        print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")
        label2metrics[phoneme] = {"mean": 0.0, "logvar": 0.0, "mse": 0.0, "n_samples": len(train_dl.dataset)}

        results = compute_mean_logvar_mse(vae=vae, X=train_dl.dataset.neural_windows)

        label2metrics[phoneme]["mean"] += results.mean
        label2metrics[phoneme]["logvar"] += results.logvar
        label2metrics[phoneme]["mse"] += results.mse

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
