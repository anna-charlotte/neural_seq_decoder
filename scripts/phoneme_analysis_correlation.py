"""Iterate over all phoneme classes, load PhonemeDataset and plot the distribution of logits, and probabilities of this phoneme."""

import pickle
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision import transforms

from data.augmentations import GaussianSmoothing
from data.dataset import (
    PhonemeDataset,
    load_averaged_windows_for_all_classes,
    save_averaged_windows_for_all_classes,
)
from evaluation import compute_cross_correlation
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import (
    DISTANCE_METRICS,
    PHONE_DEF,
    PHONE_DEF_SIL,
    ROOT_DIR,
    reorder_neural_window,
)
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from text2brain.visualization import plot_neural_window
from utils import load_pkl


def compute_distances(
    averaged: Dict[int, torch.Tensor],
    samples: Dict[int, List[torch.Tensor]],
    out_dir: Path,
    distance_metric: str,
):
    true_classes = []
    closest_classes = []
    ranks = []

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
                        f"Distance metric is invalid: {distance_metric}. Valid options are: {DISTANCE_METRICS}."
                    )
                distances.append(distance)

            # Rank the true class
            sorted_distances = sorted((dist, kls) for kls, dist in enumerate(distances))
            curr_rank = None
            for rank, (dist, kls) in enumerate(sorted_distances):
                if kls == sample_phoneme_cls:
                    curr_rank = rank
                    break
            if curr_rank is None:
                raise ValueError("No rank found!")
            else:
                ranks.append(curr_rank)

            closest_class = distances.index(min(distances))
            true_classes.append(sample_phoneme_cls)
            closest_classes.append(closest_class)


def extract_samples_for_each_label(train_file: Path, n_samples: int, classes: list, transform=None):
    print(f"Extract {n_samples} samples for each class ...")
    # with open(train_file, "rb") as handle:
    #     train_data = pickle.load(handle)
    train_data = load_pkl(train_file)

    train_dl = get_data_loader(
        data=train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=None,
        transform=transform,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": "C", "phoneme_cls": classes},
    )
    train_ds = train_dl.dataset

    samples_per_label = {idx: [] for idx in range(len(classes))}
    for sample in train_ds:
        X, label, _, _ = sample
        key = classes.index(label.item())

        if len(samples_per_label[key]) < n_samples:
            samples_per_label[key].append(X)

        if all(len(samples) >= n_samples for samples in samples_per_label.values()):
            break

    print("Done.")
    return samples_per_label


def main(args: dict) -> None:
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)

    n_classes = 39
    phoneme_classes = list(range(1, n_classes + 1))

    transform = transforms.Compose(
        [
            TransposeTransform(),
            ReorderChannelTransform(),
            AddOneDimensionTransform(dim=0),
            GaussianSmoothing(256, kernel_size=20, sigma=2.0, dim=1),
            SoftsignTransform(),
        ]
    )
    sample_windows_per_phoneme = extract_samples_for_each_label(
        train_file=args["train_set_path"],
        n_samples=10,
        classes=phoneme_classes,
        transform=transform,
    )
    print(f"len(sample_windows_per_phoneme) = {len(sample_windows_per_phoneme)}")

    X = []
    Y = []

    for class_label, samples in sample_windows_per_phoneme.items():
        X.extend(samples)
        Y.extend([class_label] * len(samples))

    # correlations
    for x1, y1 in zip(X, Y):
        for x2, y2 in zip(X, Y):
            x1 = x1.squeeze()
            x2 = x2.squeeze()
            cross_corr = compute_cross_correlation(x1, x2)
            cross_corr_same = compute_cross_correlation(x1, x1)
            print(f"cross_corr.shape = {cross_corr.shape}")
            print(f"np.amax(cross_corr) = {np.amax(cross_corr)}")
            print(f"np.amin(cross_corr) = {np.amin(cross_corr)}")

            print(f"cross_corr_same.shape = {cross_corr_same.shape}")
            print(f"np.amax(cross_corr_same) = {np.amax(cross_corr_same)}")
            print(f"np.amin(cross_corr_same) = {np.amin(cross_corr_same)}")

    # transform = transforms.Compose(
    #     [
    #         TransposeTransform(),
    #         ReorderChannelTransform(),
    #         AddOneDimensionTransform(dim=0),
    #         SoftsignTransform()
    #     ]
    # )
    # sample_windows_per_phoneme = extract_samples_for_each_label(
    #     train_file=args["train_set_path"],
    #     n_samples=100,
    #     classes=range(1, 40),
    #     transform=transform,
    # )
    # print(f"len(sample_windows_per_phoneme) = {len(sample_windows_per_phoneme)}")

    # X = []
    # Y = []

    # for class_label, samples in sample_windows_per_phoneme.items():
    #     X.extend(samples)
    #     Y.extend([class_label] * len(samples))

    # # plot different gaussian smoothings all next to each other
    # gaussian_smoothers = [
    #     GaussianSmoothing(256, kernel_size, sigma, dim=1) for kernel_size in [5, 10, 20] for sigma in [1.0, 2.0, 3.0]
    # ]

    # sample_dir = (
    #     ROOT_DIR
    #     / "plots"
    #     / "gaussian_smoothing"
    # )
    # sample_dir.mkdir(parents=True, exist_ok=True)

    # for i, image in enumerate(X):
    #     fig, axes = plt.subplots(1, len(gaussian_smoothers) + 1, figsize=(32, 8))
    #     for j, ax in enumerate(axes):
    #         print(f"image.size() = {image.size()}")
    #         if j == 0:
    #             img = image
    #             title = "Original image"
    #         else:
    #             smoother = gaussian_smoothers[j-1]
    #             img = smoother(image)
    #             title = f'Smoothed{j}\nkernel_size={smoother.kernel_size}\nsigme={smoother.sigma}'
    #         ax.imshow(img.squeeze(), cmap='plasma')
    #         ax.axis('off')
    #         ax.set_title(title)

    #     out_file = sample_dir / f"gaussian_smoothing_comparison_image_{i}.png"
    #     plt.savefig(out_file)


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"

    main(args)
