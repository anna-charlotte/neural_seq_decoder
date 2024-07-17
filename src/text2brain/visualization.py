from pathlib import Path
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from neural_decoder.neural_decoder_trainer import get_dataset_loaders
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR


def plot_neural_window(
    window: torch.Tensor, out_file, title: str, figsize: Tuple[int, int] = (10, 6)
) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(window, cmap="plasma", aspect="auto")  # , vmin=overall_min_value, vmax=overall_max_value)
    plt.colorbar()

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Channel")

    plt.savefig(out_file)
    plt.close()


def plot_brain_signal_animation(signal: torch.Tensor, save_path: Path, title: str = "Frame") -> None:
    print(signal.size())
    # assert len(signal.size()) == 2
    # assert signal.size(1) == 256

    reshaped_signal = signal.view(4, -1, 8, 8)
    img_list = []

    for i in range(reshaped_signal.size(1)):
        imgs = []
        for j, img in enumerate(reshaped_signal[:, i, :, :]):
            imgs.append(img.detach().cpu())

        img_list.append(imgs)

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    ims = []
    cbs = []

    for ax, img in zip(axs.flatten(), img_list[0]):
        im = ax.imshow(img, cmap="viridis")
        ims.append(im)
        cb = fig.colorbar(im, ax=ax)
        cbs.append(cb)

    titles = ["Signal 1 (6v)", "Signal 2 (6v)", "Spike power 1", "Spike power 2"]

    def animate(i):
        for j, (im, img, ax) in enumerate(zip(ims, img_list[i], axs.flatten())):
            im.set_data(img)

            cbs[j].remove()
            cb = fig.colorbar(im, ax=ax)
            cbs[j] = cb
            ax.set_title(titles[j])

        fig.suptitle(f"{title} {i}")

        return ax

    ani = animation.FuncAnimation(fig, animate, frames=len(img_list), interval=200, repeat=False)
    ani.save(save_path, writer="imagemagick", fps=5)


def plot_accuracies(
    accs: np.ndarray,
    out_file: Path,
    title: str = "Phoneme Classification - Test Accuracies",
    x_label: str = "Phoneme",
    y_label: str = "Test Accuracy",
) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=PHONE_DEF, y=accs, palette="muted")

    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_file)


def plot_metric_of_multiple_models(
    model_names: List[str],
    values: list,
    classes: list,
    title: str,
    out_file: Path,
    xlabel: str = "classes",
    ylabel: str = "accuracy",
) -> None:
    # Colors for the bars for each model
    colors = ["blue", "orange", "green", "red", "purple"]

    # Number of classes and models
    n_classes = len(classes)
    n_models = len(model_names)

    # Set the positions of the bars on the x-axis
    bar_width = 0.8 / n_models  # Adjust bar width based on the number of models
    index = np.arange(n_classes)

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, vals) in enumerate(zip(model_names, values)):
        ax.bar(index + i * bar_width, vals, bar_width, label=model, color=colors[i % len(colors)])

    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(classes)
    ax.legend()

    # Adding a grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Display the plot
    plt.tight_layout()
    print(f"Saving plot to: {out_file}")
    plt.savefig(out_file)


def plot_phoneme_distribution(
    class_counts: List[int],
    out_file: Path,
    title: str = "Phoneme Class Counts",
    x_label: str = "Phoneme",
    y_label: str = "Count",
) -> None:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=PHONE_DEF, y=class_counts, palette="muted")

    max_height = max(class_counts)
    ax.set_ylim(0, 1.15 * max_height)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 18),  # Move the text 12 points above the bar
            textcoords="offset points",
            rotation=90,
        )  # Rotate the text by 90 degrees

    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    print(f"Save distribution plot to: {out_file}")
    plt.savefig(out_file)


if __name__ == "__main__":
    dataset_path = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    print(ROOT_DIR)
    save_path = ROOT_DIR / "plots" / "data_visualization" / "animation.gif"
    print(save_path)

    train_dl, test_dl, loaded_data = get_dataset_loaders(dataset_path, batch_size=1)

    for i, batch in enumerate(train_dl):
        X, y, _, _, _ = batch
        for signal in X:
            plot_brain_signal_animation(signal=signal, save_path=save_path)
            break
        break
