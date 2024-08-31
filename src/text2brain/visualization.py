from pathlib import Path
from typing import Dict, List, Tuple
import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from neural_decoder.neural_decoder_trainer import get_dataset_loaders
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from evaluation import compute_auroc_with_stderr


def plot_aurocs_with_error_bars(aurocs, errs, x_labels, out_file: Path, title="AUROC Plot", ylabel="AUROC", xlabel="Models", color='blue'):

    aurocs = np.array(aurocs)
    errs = np.array(errs)
    x_labels = [str(x) for x in x_labels]

    plt.figure(figsize=(10, 6))
    
    plt.bar(x_labels, aurocs, yerr=errs, capsize=5, color="skyblue", edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    range_auroc = max(aurocs) - min(aurocs)
    min_padding = range_auroc / 3
    max_padding = range_auroc / 4
    plt.ylim(min(aurocs) - min_padding, max(aurocs) + max_padding)

    # plt.xticks(ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def bar_plot(bars, x_labels, out_file: Path, title="AUROC Plot", ylabel="AUROC", xlabel="Models", color='blue'):

    print(f"bars = {bars}")
    bars = np.array(bars)
    print(f"bars.shape = {bars.shape}")
    x_labels_new = copy.deepcopy(x_labels)
    for i, x in enumerate(x_labels):
        x_labels_new[i] = f"{x} \n(val={bars[i]:.4f})"

    plt.figure(figsize=(10, 6))
    
    plt.bar(x_labels_new, bars, capsize=5, color="skyblue", edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    diff = max(bars) - min(bars)
    min_padding = diff / 3
    max_padding = diff / 4
    plt.ylim(min(bars) - min_padding, max(bars) + max_padding)

    # plt.xticks(ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


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


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    out_file: Path,
    xlabel: str,
    ylabel: str,
    title: str = "Correlation matrix",
):
    # Plot the heatmap
    plt.figure(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=False, fmt="d", cmap="viridis")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)

    print(f"Save correlation matrix plot to: {out_file}")
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


def plot_tsne(vectors: List[np.ndarray], labels: List[int], title: str, out_file: Path) -> None:
    stacked_vectors = np.concatenate(vectors, axis=0)

    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(stacked_vectors)

    labels = np.array(labels)
    unique_classes = np.unique(labels)

    plt.figure(figsize=(10, 10))
    for cls in unique_classes:
        idx = labels == cls
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], alpha=0.6, s=3, label=f'Class {PHONE_DEF[cls - 1]}')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend(title="Phoneme classes")

    print(f"Save tsne plot to: {out_file}")
    plt.savefig(out_file)


def plot_means_and_stds(means: np.ndarray, stds: np.ndarray, phoneme: str, out_dir: Path = None):
    if out_dir is None:
        out_dir = ROOT_DIR / "evaluation" / "vae" / "latent_dim_evaluation"

    # FIRST PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    n_dim = len(means)
    ax1.hist(means, bins=20, color="blue", edgecolor="black")
    ax1.set_title(f"Histogram of Mean Values ({n_dim} dim)")
    ax1.set_xlabel("Mean value of each channel in the latent space over all encodings")
    ax1.set_ylabel("Frequency")

    ax2.hist(stds, bins=20, color="green", edgecolor="black")
    ax2.set_title(f"Histogram of Standard Deviation (STD) Values ({n_dim} dim))")
    ax2.set_xlabel("STD value of each channel in the latent space over all encodings")
    ax2.set_ylabel("Frequency")

    fig.suptitle(f"VAE, Average Means and Standard Deviations (phonemes {phoneme})", fontsize=16)
    plt.tight_layout()

    out_file = (
        out_dir
        / f"average_means_and_stds__phoneme_{phoneme}__histogram.png"
    )
    plt.savefig(out_file)
    print(f"Saved plot to: {out_file}")

    # SECOND PLOT
    channels = list(range(1, n_dim + 1))

    plt.figure(figsize=(12, 6))
    plt.scatter(channels, means, color="blue", label="Mean Values", marker="x")
    plt.scatter(channels, stds, color="green", label="STD Values", marker="o")

    plt.title("Mean and STD Values for Each Channel")
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.legend()

    out_file = (
        out_dir
        / f"average_means_and_stds__phoneme_{phoneme}__per_channel.png"
    )
    plt.savefig(out_file)
    print(f"Saved plot to: {out_file}")



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


def plot_original_vs_reconstructed_image(X: np.ndarray, X_recon: np.ndarray, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))

    # set the color range
    vmin, vmax = -1, 1

    # display the first image
    im0 = axes[0].imshow(X, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[0].axis("off")
    axes[0].set_title("Original image")

    # display the second image
    im1 = axes[1].imshow(X_recon, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].axis("off")
    axes[1].set_title("Reconstructed image")

    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_single_image(X: np.ndarray, out_file: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vmin, vmax = -1, 1

    im0 = ax.imshow(X, cmap="plasma", vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_title(title)

    fig.colorbar(im0, ax=ax)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



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




