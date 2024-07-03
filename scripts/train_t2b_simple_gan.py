import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.visualization import plot_brain_signal_animation


def main(args: dict) -> None:

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)

    device = args["device"]
    batch_size = args["batch_size"]
    phoneme_cls = [i for i in range(41)]
    print(f"device = {device}")

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    filter_by = {}
    if isinstance(phoneme_cls, int):
        filter_by = {"phoneme_cls": phoneme_cls}

    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=filter_by,
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
        phoneme_ds_filter=filter_by,
    )

    n_channels = 32
    latent_dim = 100

    ngf = 64
    ndf = 64

    n_epochs = 50
    lr = 0.0001
    clip_value = 0.01
    n_critic = 15

    gan = PhonemeImageGAN(
        latent_dim=latent_dim,
        phoneme_cls=phoneme_cls,
        n_channels=n_channels,
        ndf=ndf,
        ngf=ngf,
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )

    G_losses = []
    D_losses = []
    noise_vector = torch.randn(1, gan.g.latent_dim, device=gan.device)

    for epoch in range(n_epochs):
        for i, data in enumerate(train_dl):
            errD, errG = gan(data)

            # output training stats
            if i % 250 == 0:
                print(
                    f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {errD.item()} Loss_G: {errG.item()}"
                )

            #     phoneme = 2
            #     signal = gan.g(noise_vector, torch.tensor([phoneme]).to(device))
            #     plot_brain_signal_animation(
            #         signal=signal,
            #         save_path=ROOT_DIR
            #         / "plots"
            #         / "data_visualization"
            #         / f"phone_{phoneme}_FAKE_ep{epoch}_i_{i}.gif",
            #         title=f"Phoneme {phoneme}, Frame",
            #     )

            # if i == 0:
            #     X, _, _, _ = data

            #     phoneme = 2
            #     for j in range(X.size(0)):
            #         sub_signal = X[j, :, :]
            #         print(sub_signal.size())
            #         plot_brain_signal_animation(
            #             signal=sub_signal,
            #             save_path=ROOT_DIR
            #             / "plots"
            #             / "data_visualization"
            #             / f"phone_{phoneme}_REAL_sample_{j}.gif",
            #             title=f"Real sample, Phoneme {phoneme}, Frame",
            #         )

            # save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    plot_gan_losses(G_losses, D_losses, out_file=ROOT_DIR / "plots" / "data_visualization" / "gan_losses.png")


def plot_gan_losses(g_losses: list, d_losses: list, out_file: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_file)


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    args[
        "train_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    args[
        "test_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 8
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    main(args)
