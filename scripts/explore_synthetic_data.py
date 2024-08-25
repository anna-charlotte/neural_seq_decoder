
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


def plot_grid_comparison_of_ds_images(
        datasets: List[Union[PhonemeDataset, SyntheticPhonemeDataset]], 
        dataset_names: List[str], 
        classes: List[int], 
        out_file: Path, 
        pad: int = 3
    ) -> None:
    fig, axes = plt.subplots(len(datasets), len(classes), figsize=(10, 10), constrained_layout=True)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    for i, dataset in enumerate(datasets):

        for j, cls in enumerate(classes):
            class_indices = [idx for idx, data in enumerate(dataset) if data[1] == cls]
            class_indices = random.sample(class_indices, 30)

            padding = (pad, pad, pad, pad) 
            images = [
                F.pad(dataset[idx][0].squeeze(), padding, mode='constant', value=100) 
                for idx in class_indices
            ]

            row_1 = torch.cat(images[:15], dim=1)
            row_2 = torch.cat(images[15:], dim=1)
            concatenated_image = torch.cat([row_1, row_2], dim=0)
            
            concatenated_image_np = concatenated_image.numpy()
            masked_image = np.ma.masked_where(concatenated_image_np == 100, concatenated_image_np)
           

            cmap = plt.get_cmap('plasma')
            cmap.set_bad(color='white')

            axes[i, j].imshow(masked_image, cmap='plasma', vmin=-1., vmax=1.)
            axes[i, j].axis('off')
            
            if i == 0:
                axes[i, j].set_title(f'Class {cls}', fontsize=20)
                
    for i, dataset_name in enumerate(dataset_names):
        fig.text(0.05, 1 - (i + 0.5) / len(datasets), dataset_name, va='center', ha='center', rotation=90, size='large')

    plt.tight_layout()
    plt.savefig(out_file)


def plot_pixel_distributions(
    datasets: List[Union[PhonemeDataset, SyntheticPhonemeDataset]], 
    dataset_names: List[str],
    out_file: Path,
) -> None:
    sns.set(style="whitegrid")   # Use seaborn for a nicer style
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 10), constrained_layout=True, sharey=True)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    colors = sns.color_palette("Set2", len(datasets))  # Use a seaborn color palette

    for i, dataset in enumerate(datasets):
        pixel_values = torch.cat([sample[0].squeeze().view(-1) for sample in dataset])
        pixel_values_np = pixel_values.numpy()

        counts, bins, patches = axes[i].hist(pixel_values_np, bins=50, density=True, color=colors[i], alpha=0.7, edgecolor='black')

        bin_width = bins[1] - bins[0]
        percentages = (counts * bin_width) * 100

        axes[i].clear()
        axes[i].bar(bins[:-1], percentages, width=bin_width, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

        axes[i].set_xlim([-1, 1])
        axes[i].set_ylim([0, max(percentages) * 1.1])

        axes[i].set_title(f'Pixel Distribution for {dataset_names[i]}', fontsize=16, fontweight='bold')
        axes[i].set_xlabel('Pixel Value', fontsize=14)
        axes[i].set_ylabel('Percentage (%)', fontsize=14)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300) 



def plot_pixel_distributions_2(
    datasets: List[Union[PhonemeDataset, SyntheticPhonemeDataset]], 
    dataset_names: List[str],
    classes: List[int], 
    out_file: Path,
) -> None:
    sns.set(style="whitegrid")  # Use seaborn for a nicer style
    num_datasets = len(datasets)
    num_classes = len(classes)
    
    fig, axes = plt.subplots(num_datasets, num_classes, figsize=(15, 5 * num_datasets), constrained_layout=True, sharey=True)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    colors = sns.color_palette("Set2", num_datasets)  # Use a seaborn color palette

    for i, dataset in enumerate(datasets):
        for j, cls in enumerate(classes):
            # Filter the dataset for the current class
            pixel_values = torch.cat([sample[0].squeeze().view(-1) for sample in dataset if sample[1] == cls])
            pixel_values_np = pixel_values.numpy()

            counts, bins, patches = axes[i, j].hist(pixel_values_np, bins=50, density=True, color=colors[i], alpha=0.7, edgecolor='black')

            bin_width = bins[1] - bins[0]
            percentages = (counts * bin_width) * 100

            axes[i, j].clear()
            axes[i, j].bar(bins[:-1], percentages, width=bin_width, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

            axes[i, j].set_xlim([-1, 1])
            axes[i, j].set_ylim([0, max(percentages) * 1.1])

            if i == 0:
                axes[i, j].set_title(f'Class {cls}', fontsize=16, fontweight='bold')
            axes[i, j].set_xlabel('Pixel Value', fontsize=14)
            if j == 0:
                axes[i, j].set_ylabel(f'{dataset_names[i]}\nPercentage (%)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300) 



def main(args: dict) -> None:
    print("Exploring synthetic data vs. real data")

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

    n_samples = 10_000

    # load generative models
    gan = load_t2b_gen_model(weights_path=args["gan_weights_path"])
    vae = load_t2b_gen_model(weights_path=args["vae_weights_path"])
    
    gan_ds = gan.create_synthetic_phoneme_dataset(
        n_samples=n_samples,
        neural_window_shape=(1, 256, 32),
    )
    vae_ds = vae.create_synthetic_phoneme_dataset(
        n_samples=n_samples,
        neural_window_shape=(1, 256, 32),
    )

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
    real_ds = train_dl_real.dataset

    datasets = [real_ds, gan_ds, vae_ds]
    dataset_names = ['Real Dataset', 'GAN Dataset', 'VAE Dataset']

    phoneme_classes = args["phoneme_cls"]


    # plot random images, real vs GAN generated vs VAE generated 
    pad = 3
    file = ROOT_DIR / "plots" / f"synthetic_images_comparison_{pad}.png"
    plot_grid_comparison_of_ds_images(datasets, dataset_names, phoneme_classes, out_file=file, pad=pad)

    # plot pixel distributions
    file = ROOT_DIR / "plots" / f"synthetic_images_pixel_distributions_entire_dataset.png"
    plot_pixel_distributions(datasets, dataset_names, out_file=file)




if __name__ == "__main__":
    print("in main ...")

    args = {}
    args["seed"] = 2
    args["transform"] = "softsign"
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    args["vae_weights_path"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110"  # cls [3, 31]
    args["gan_weights_path"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{1000}"
    
    args["gaussian_smoothing_kernel_size"] = 20
    args["gaussian_smoothing_sigma"] = 2.0
    args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
    args["correctness_value"] = ["C"]

    main(args)
