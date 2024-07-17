import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.visualization import plot_brain_signal_animation
from utils import set_seeds


def main(args: dict) -> None:
    set_seeds(args["seed"])
    torch.use_deterministic_algorithms(True)

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args["device"]
    batch_size = args["batch_size"]
    phoneme_cls = [10, 22]  # list(range(1, 40))
    args["phoneme_cls"] = phoneme_cls

    train_file = args["train_set_path"]
    with open(train_file, "rb") as handle:
        train_data = pickle.load(handle)

    transform = None
    if args["transform"] == "softsign":
        transform = SoftsignTransform()

    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": phoneme_cls}
    if isinstance(phoneme_cls, int):
        phoneme_ds_filter["phoneme_cls"] = phoneme_cls
    args["phoneme_ds_filter"] = phoneme_ds_filter

    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
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
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )

    gan = PhonemeImageGAN(
        latent_dim=args["latent_dim"],
        phoneme_cls=args["phoneme_cls"],
        n_channels=args["n_channels"],
        ndf=args["ndf"],
        ngf=args["ngf"],
        device=args["device"],
        n_critic=args["n_critic"],
        clip_value=args["clip_value"],
        lr=args["lr"],
    )

    # load or compute the average images
    print(f"out_dir = {out_dir}")

    G_losses = []
    D_losses = []
    # noise_vector = torch.randn(1, gan._g.latent_dim, device=gan.device)
    n_epochs = args["n_epochs"]

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    for epoch in range(n_epochs):
        print(f"epoch = {epoch}")
        for i, data in enumerate(train_dl):
            errD, errG = gan(data)

            # output training stats
            if i % 250 == 0:
                print(
                    f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {errD.item()} Loss_G: {errG.item()}"
                )

            #     phoneme = 2
            #     signal = gan._g(noise_vector, torch.tensor([phoneme]).to(device))
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

        # save GAN
        file = out_dir / f"modelWeights_epoch_{epoch}"
        print(f"Storing GAN weights to: {file}")
        gan.save_state_dict(file)
        # torch.save(gan.state_dict(), file)

    plot_gan_losses(
        G_losses,
        D_losses,
        out_file=ROOT_DIR / "plots" / "data_visualization" / f"gan_losses_nclasses_{len(phoneme_cls)}.png",
    )


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
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"

    args["n_classes"] = 2

    args["batch_size"] = 16
    # args["n_output_features"] = 256
    # args["hidden_dim"] = 512
    # args["n_layers"] =

    args["n_channels"] = 128
    args["latent_dim"] = 100

    args["ngf"] = 64
    args["ndf"] = 64

    args["n_epochs"] = 50
    args["lr_generator"] = 0.0001
    args["lr_discriminator"] = 0.0001
    args["lr"] = 0.0001
    args["clip_value"] = 0.01
    args["n_critic"] = 1
    args["transform"] = "softsign"

    args[
        "train_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    args[
        "test_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    args[
        "output_dir"
    ] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_{timestamp}__nclasses_{args['n_classes']}"

    main(args)
