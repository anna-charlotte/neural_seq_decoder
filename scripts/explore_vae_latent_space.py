
from datetime import datetime
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from typing import List, Union
from pathlib import Path

from torchvision import transforms
import random

from data.augmentations import GaussianSmoothing
from data.dataset import PhonemeDataset, SyntheticPhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from text2brain.models.model_interface_load import load_t2b_gen_model
from text2brain.models.vae import CondVAE, VAE
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from utils import load_pkl, set_seeds

from text2brain.visualization import  plot_tsne


def main(args: dict) -> None:
    print("VAE latent space visualization ...")

    set_seeds(args["seed"])
    device = args["device"]

    phoneme_ds_filter = {"correctness_value": args["correctness_value"], "phoneme_cls": args["phoneme_cls"]}
    args["phoneme_ds_filter"] = phoneme_ds_filter

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

    vae = load_t2b_gen_model(weights_path=args["vae_weights_path"])
    
    # load real val set
    train_dl_real = get_data_loader(
        data=load_pkl(args["train_set_path"]),
        batch_size=1,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=None,
        transform=transform,
    )

    # VAE latent space visualization
    vae_mus = []
    vae_labels = []
    for i, batch in enumerate(train_dl_real):
        # if i == 1_000:
        #     break

        X, y, _, _ = batch
        X = X.to(device)
        
        mu, _ = vae.encode(X)
        latent_dim = mu.size(1)
        mu = mu.detach().cpu().numpy()
        vae_mus.append(mu)
        vae_labels.append(y.item())


    # vae_encs_concat = torch.cat(vae_mus, dim=0)
    file = ROOT_DIR / "plots" / "tsne" / f"tsne_vae_latent_space_mus_all_{args['timestamp']}__latent_dim_{latent_dim}.png"
    if isinstance(vae, VAE):
        title = f'TSNE of latent space of unconditional VAE \n(latent_dim={latent_dim}, trained on classes: {vae.classes})'
    elif isinstance(vae, CondVAE):
        title = f'TSNE of latent space of conditional VAE \n(latent_dim={latent_dim}, trained on classes: {vae.classes})'
    plot_tsne(vectors=vae_mus, labels=vae_labels, title=title, out_file=file)




if __name__ == "__main__":
    print("in main ...")

    args = {}
    args["seed"] = 2
    args["transform"] = "softsign"
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")

    for latent_dim in [32, 64, 128, 256, 512]:
        for epoch in [200]:
            args["vae_weights_path"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_{latent_dim}/modelWeights_epoch_{epoch}"
            
            args["gaussian_smoothing_kernel_size"] = 20
            args["gaussian_smoothing_sigma"] = 2.0
            args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
            args["correctness_value"] = ["C"]
            
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            args["timestamp"] = timestamp

            main(args)
