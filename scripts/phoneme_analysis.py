"""Iterate over all phoneme classes, load PhonemeDataset and plot the distribution of logits, and probabilities of this phoneme."""

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import (
    PHONE_DEF,
    PHONE_DEF_SIL,
    ROOT_DIR,
    reorder_neural_window,
    load_averaged_windows_for_all_classes,
    save_averaged_windows_for_all_classes
)

DISTANCE_METRICS = ["frobenius", "cosine_sim", "manhattan", "mse"]


def softmax(logits):
    logits_tensor = torch.tensor(logits)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=0)
    return probabilities


def plot_heatmap(
    true_classes: List[int],
    closest_classes: List[int],
    n_classes: int,
    out_file: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: list,
    yticks: list,
):
    # Create confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for closest, true in zip(closest_classes, true_classes):
        confusion_matrix[closest, true] += 1

    print()
    # Print the sum of each row in the confusion matrix
    for i, row in enumerate(confusion_matrix):
        row_sum = np.sum(row)
        print(f"Sum of row {i}: {row_sum}")

    correctly_classified = np.trace(confusion_matrix)
    total_sum = np.sum(confusion_matrix)
    print(f"\nCorrectly classified by distance measure: {correctly_classified} / {total_sum}")

    # Plot the heatmap
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix, annot=False, fmt="d", cmap="viridis", xticklabels=xticks, yticklabels=yticks
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add title and subtitle
    plt.suptitle(title, fontsize=16)
    plt.title(
        f"(Total number of samples: {total_sum}, Correctly classified by distance metric: {correctly_classified})",
        fontsize=12,
        pad=20,
    )
    print(f"Save heatmap plot to: {out_file}")
    plt.savefig(out_file)


def compute_distances(
    averaged: Dict[int, torch.Tensor],
    samples: Dict[int, List[torch.Tensor]],
    distance_metric: str,
    out_dir: Path = None,
    verbose: bool = False
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
                    curr_dist = dist
                    break
            if curr_rank is None:
                raise ValueError("No rank found!")
            else:
                if verbose:
                    print(f"curr_rank = {curr_rank}")
                    print(f"curr_dist = {curr_dist}")
                ranks.append(curr_rank)

            closest_class = distances.index(min(distances))
            true_classes.append(sample_phoneme_cls)
            closest_classes.append(closest_class)

    if out_dir is not None:
        plot_heatmap(
            true_classes,
            closest_classes,
            n_classes=len(averaged),
            out_file=out_dir / f"true_vs_closest_classes__{distance_metric}.png",
            title=f"Heatmap of True Classes vs Closest Classes (distance: {distance_metric})",
            xlabel="True Classes",
            ylabel="Closest Classes",
            xticks=PHONE_DEF,
            yticks=PHONE_DEF,
        )

        plot_heatmap(
            true_classes,
            ranks,
            n_classes=len(averaged),
            out_file=out_dir / f"true_vs_ranks__{distance_metric}.png",
            title=f"Heatmap of True Classes vs rank (distance: {distance_metric})",
            xlabel="True classes",
            ylabel="Rank of the distance to the true class",
            xticks=PHONE_DEF,
            yticks=range(1, len(averaged) + 1),
        )


def extract_samples_for_each_label(train_file: Path, n_samples: int, classes: list, reorder_channels: bool):
    print(f"Extract {n_samples} samples for each class ...")
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    train_dl = get_data_loader(
        data=train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": "C", "phoneme_cls": classes},
    )
    train_ds = train_dl.dataset

    samples_per_label = {idx: [] for idx in range(len(classes))}
    for sample in train_ds:
        X, label, _, _ = sample
        X = torch.transpose(X, 0, 1)
        key = classes.index(label.item())
        if len(samples_per_label[key]) < n_samples:
            if reorder_channels:
                X = reorder_neural_window(X)

            samples_per_label[key].append(X)
        if all(len(samples) >= n_samples for samples in samples_per_label.values()):
            break

    # Convert defaultdict to a regular dictionary for ease of use
    samples_per_label = dict(samples_per_label)
    print("Done.")
    return samples_per_label


def plotneural_window(window, out_file, title):
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

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)

    n_classes = 39
    phoneme_classes = list(range(1, n_classes + 1))

    device = args["device"]
    batch_size = args["batch_size"]
    all_averaged_windows = {}
    sample_windows_per_phoneme = {i: [] for i in range(n_classes)}

    avg_window_file = ROOT_DIR / "evaluation" / "phoneme_class_to_average_window_with_reordering.pt"
    # save_averaged_windows_for_all_classes(
    #     train_file=args["train_set_path"],
    #     reorder_channels=True,
    #     out_file=avg_window_file
    # )
    phoneme2avg_window = load_averaged_windows_for_all_classes(avg_window_file)

    overall_min_value = float("inf")
    overall_max_value = -float("inf")

    for phoneme_cls in phoneme_classes:
        avg_window = phoneme2avg_window[phoneme_cls]["avg_window"]

        min_value = torch.min(avg_window).item()

        if min_value < overall_min_value:
            overall_min_value = min_value
        max_value = torch.max(avg_window).item()
        if max_value > overall_max_value:
            overall_max_value = max_value

    print(f"overall_min_value = {overall_min_value}")
    print(f"overall_max_value = {overall_max_value}")

    # plot average windows
    print(f"Plotting the average windows ...")
    for phoneme_cls in phoneme_classes:
        avg_window = phoneme2avg_window[phoneme_cls]["avg_window"]
        n_samples = phoneme2avg_window[phoneme_cls]["n_samples"]

        phoneme_name = f"{phoneme_cls}" if phoneme_cls == 0 else f'"{PHONE_DEF_SIL[phoneme_cls-1]}" ({phoneme_cls})'
        title = f"Averaged neural window for phoneme class {phoneme_name} (averaged over {n_samples} samples)"
        out_file = ROOT_DIR / "plots" / "averaged_windows_per_phoneme_with_channel_reordering__different_ranges_on_plots" / f"averaged_window_phoneme_{phoneme_cls}.png"
        plot_neural_window(avg_window, out_file, title)

    print("Done.")

    sample_windows_per_phoneme = extract_samples_for_each_label(
        args["train_set_path"], 1600, phoneme_classes, reorder_channels=True
    )

    # label index to windows
    all_averaged_windows = {i: phoneme2avg_window[p]["avg_window"] for i, p in enumerate(phoneme_classes)}

    print(f"Plot distance metric plots ...")
    for dist_metric in DISTANCE_METRICS:
        compute_distances(
            all_averaged_windows,
            sample_windows_per_phoneme,
            out_dir=ROOT_DIR
            / "plots"
            / "averaged_windows_per_phoneme_with_channel_reordering__different_ranges_on_plots",
            distance_metric=dist_metric,
        )
    print("Done.")

    # # plot logits and probailities for the phonemes
    # for phoneme_cls in phoneme_classes:
    #     print(f"phoneme_cls = {phoneme_cls}")
    #     print(f"device = {device}")

    #     train_file = args["train_set_path"]
    #     with open(train_file, "rb") as handle:
    #         train_data = pickle.load(handle)

    #     filter_by = {}
    #     if isinstance(phoneme_cls, int):
    #         filter_by = {"correctness_value": "C", "phoneme_cls": [phoneme_cls]}

    #     train_dl = get_data_loader(
    #         data=train_data,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         collate_fn=None,
    #         dataset_cls=PhonemeDataset,
    #         phoneme_ds_filter=filter_by,
    #     )
    #     train_ds = train_dl.dataset

    #     print(f"len(train_ds) = {len(train_ds)}")
    #     all_logits_for_phoneme = []
    #     all_probabilities = []

    #     for i, batch in enumerate(train_ds):
    #         X, _, logits, _ = batch
    #         window = torch.transpose(X, 0, 1)
    #         window = reorder_neural_window(window)

    #         probabilities = softmax(logits)
    #         assert phoneme_cls == torch.argmax(logits).item()

    #         logit = logits[phoneme_cls].item()
    #         all_logits_for_phoneme.append(logit)

    #         prob = probabilities[phoneme_cls]
    #         all_probabilities.append(prob)

    #     # plot distribution over logit values
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(all_logits_for_phoneme, bins=40, edgecolor="black")
    #     plt.title(
    #         f"Distribution of Logit Values (B2T RNN) for Phoneme {phoneme_cls} (total counts = {len(all_logits_for_phoneme)})"
    #     )
    #     plt.xlabel("Logit Values")
    #     plt.ylabel("Frequency")
    #     plt.xlim(left=0)
    #     plt.xlim(right=1)
    #     plt.grid(True)

    #     plt.savefig(ROOT_DIR / "plots" / "phoneme_distr_logits" / f"logits_phoneme_{phoneme_cls}.png")

    #     # plot distribution over probability values
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(all_probabilities, bins=40, edgecolor="black")
    #     plt.title(
    #         f"Distribution of Probabilitiy Values (B2T RNN) for Phoneme {phoneme_cls} (total counts = {len(all_probabilities)})"
    #     )
    #     plt.xlabel("Probabilities Values")
    #     plt.ylabel("Frequency")
    #     plt.xlim(left=0)
    #     plt.xlim(right=1)
    #     plt.grid(True)

    #     plt.savefig(ROOT_DIR / "plots" / "phoneme_distr_probabilities" / f"prob_phoneme_{phoneme_cls}.png")

    #     plt.close()


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 8
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    main(args)
