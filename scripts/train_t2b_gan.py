import json
import pickle
from datetime import datetime
from pathlib import Path
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data.dataset import PhonemeDataset
from data.augmentations import GaussianSmoothing
from data.dataset import PhonemeDataset
from neural_decoder.model_phoneme_classifier import train_phoneme_classifier
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models.model_interface_load import load_t2b_gen_model
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.visualization import plot_brain_signal_animation, plot_single_image, bar_plot
from utils import dump_args, load_args, load_pkl, set_seeds



def main(args: dict) -> None:
    for k, v in args.items():
        print(f"{k}: {v}")

    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    print(f"out_dir = {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    device = args["device"]
    batch_size = args["batch_size"]

    train_data = load_pkl(args["train_set_path"])
    transform = None
    if args["transform"] == "softsign":
        transform = transforms.Compose(
            [
                TransposeTransform(0, 1),
                ReorderChannelTransform(),
                AddOneDimensionTransform(dim=0),
                GaussianSmoothing(
                    256,
                    kernel_size=20,
                    sigma=2.0,
                    dim=1,
                ),
                SoftsignTransform(),
            ]
        )

    classes = args["phoneme_cls"]
    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": classes}

    train_dl = get_data_loader(
        data=train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )
    train_dl.name = "train-dl-real"

    print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")

    val_data = load_pkl(args["val_set_path"])
    val_dl = get_data_loader(
        data=val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )
    val_dl.name = "val-dl-real"

    test_data = load_pkl(args["test_set_path"])
    test_dl = get_data_loader(
        data=test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        transform=transform,
    )
    test_dl.name = "test-dl-real"

    gan = PhonemeImageGAN(
        input_shape=args["input_shape"],
        latent_dim=args["latent_dim"],
        classes=args["phoneme_cls"],
        conditioning=args["conditioning"],
        dec_emb_dim=args["dec_emb_dim"],
        device=device,
        n_critic=args["n_critic"],
        lr_g=args["lr_g"],
        lr_d=args["lr_d"],
        cond_bn=args["cond_bn"]
    )
    args["model_class"] = gan.__class__.__name__

    if "vae_path" in args.keys():
        print("Loading VAE.decoder state dict weights for GAN.generator")
        vae = load_t2b_gen_model(weights_path=args["vae_path"])

        vae_fc_state_dict = vae.decoder.fc.state_dict()
        gan._g.fc.load_state_dict(vae_fc_state_dict)
        vae_model_state_dict = vae.decoder.model.state_dict()
        gan._g.model.load_state_dict(vae_model_state_dict)

    print(f"Reinitializing optimizers")
    gan.init_optimizers()
    print(f"gan = {gan}")

    plot_dir = Path(args["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot_dir = {plot_dir}")

    dump_args(args, out_dir / "args")
    dump_args(args, out_dir / "args.json")
    dump_args(args, plot_dir / "args.json")

    G_losses = []
    D_losses = []

    val_aurocs = []
    test_aurocs = []
    val_f1s = []
    test_f1s = []
    epochs = []

    noise = torch.randn(3, gan._g.latent_dim, device=gan.device)

    n_epochs = args["n_epochs"]
    n_steps = int(len(train_dl) / (n_critic + 1))

    print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")
    print(f"len(train_dl) = {len(train_dl)}")
    print(f"n_steps = {n_steps}")

    highest_auroc = -1.0
    highest_f1 = -1.0

    best_auroc_file = None
    best_f1_file = None

    for epoch in range(n_epochs):
        G_losses_epoch = []
        D_losses_epoch = []

        print(f"epoch = {epoch}")
        for i in range(n_steps):
            if (i % 20 == 0 and i < 100 and epoch == 0) or (i == 0 and epoch % 5 == 0):
                # plot a single image and save to evaluation folder
                for label in classes:
                    for j in range(noise.size(0)):
                        label_t = torch.tensor([label]).to(device)

                        # generated_image = gan.generate(label=label_t).cpu().detach()
                        generated_image = gan.generate_from_given_noise(noise=noise[j].unsqueeze(0), label=label_t).cpu().detach()
                        
                        plot_single_image(
                            X=generated_image[0][0],
                            out_file=plot_dir / f"generated_image_{epoch}_{i}_{j}__cls_{label}.png",
                            title=f"GAN generated image \n(phoneme cls {label}, epoch {epoch})"
                        )


            loss_D, loss_G = gan.train_step(train_dl)

            # save losses for plotting
            G_losses_epoch.append(loss_G.item())
            D_losses_epoch.append(loss_D.item())
        
        avg_G_loss = sum(G_losses_epoch) / len(G_losses_epoch)
        avg_D_loss = sum(D_losses_epoch) / len(D_losses_epoch)
        
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)

        # output training stats
        print(
            f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {avg_D_loss} Loss_G: {avg_G_loss}"
        )

        # save GAN
        file = out_dir / f"modelWeights"
        print(f"Storing GAN weights to: {file}")

        if epoch > 650 and epoch % 20 == 0:

            if len(classes) > 1:
                stats = train_phoneme_classifier(
                    gen_models=[gan],
                    n_samples_train_syn=10_000,
                    n_samples_val=5_000,
                    n_samples_test=5_000,
                    val_dl_real=val_dl, 
                    test_dl_real=test_dl,
                    phoneme_classes=classes,
                    input_shape=(128, 8, 8),
                    batch_size=64,
                    n_epochs=100, 
                    patience=40,
                    lr=1e-4,
                    device=device,
                    out_dir=out_dir,
                )

                val_auroc = max(stats["val_aurocs_macro"]["val-dl-real"])
                if val_auroc > highest_auroc:
                    highest_auroc = val_auroc

                    if best_auroc_file is not None and os.path.exists(best_auroc_file):
                        os.remove(best_auroc_file)
                    
                    best_auroc_file = out_dir / f"modelWeights_{epoch}_best_auroc"
                    gan.save_state_dict(best_auroc_file)

                val_f1 = max(stats["val_f1_macro"]["val-dl-real"])
                if val_f1 > highest_f1:
                    highest_f1 = val_f1

                    if best_f1_file is not None and os.path.exists(best_f1_file):
                        os.remove(best_f1_file)
                    
                    best_f1_file = out_dir / f"modelWeights_{epoch}_best_f1"
                    gan.save_state_dict(best_f1_file)


            else:
                file = out_dir / f"modelWeights_{epoch}"
                gan.save_state_dict(file)

        plot_gan_losses(
            G_losses,
            D_losses,
            out_file=plot_dir / f"gan_losses_nclasses_{len(classes)}.png",
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


def plot_accuracies(accs: list, out_file: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(accs, label="G")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(out_file)


if __name__ == "__main__":

    for seed in [8, 9,]:
        for lr_g, lr_d in [(1e-4, 5e-5),]:  #, (1e-5, 1e-5), (1e-4, 1e-5)]:  # [(5e-05, 5e-05), (1e-04, 1e-05), (1e-05, 1e-05)]:  #, 1e-5, 1e-3]:  # [1e-3, 1e-4, 1e-5]:
            for n_critic in [5]:  # , 2]:
                for latent_dim in [512]:
                    for phoneme_cls in [[3, 31],]:
                        for conditioning in ["concat"]:
                            for dec_emb_dim in [32]:

                                now = datetime.now()
                                timestamp = now.strftime("%Y%m%d_%H%M%S")

                                print("in main ...")
                                args = {}
                                args["seed"] = seed
                                args["device"] = "cuda"
                                args["timestamp"] = timestamp

                                args["phoneme_cls"] = phoneme_cls
                                args["conditioning"] = conditioning
                                args["cond_bn"] = True

                                args["batch_size"] = 128

                                args["latent_dim"] = latent_dim
                                args["dec_emb_dim"] = dec_emb_dim
                                args["input_shape"] = (4, 64, 32)

                                args["n_epochs"] = 1800
                                args["lr_g"] = lr_g
                                args["lr_d"] = lr_d
                                
                                args["n_critic"] = n_critic
                                args["transform"] = "softsign"

                                args["train_set_path"] = (
                                    "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
                                )
                                args["val_set_path"] = (
                                    "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl"
                                )
                                args["test_set_path"] = (
                                    "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl"
                                )
                                args["output_dir"] = (
                                    f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/gan_conditioning/ld_{latent_dim}/gan__conditioning_{conditioning}__dec_emb_dim_{dec_emb_dim}__phoneme_cls_{'_'.join(map(str, args['phoneme_cls']))}__seed_{seed}"
                                )
                                args["plot_dir"] = str(
                                    ROOT_DIR
                                    / "evaluation"
                                    / "experiments"
                                    / "gan_conditioning"
                                    / f"ld_{latent_dim}"
                                    / f"gan__conditioning_{conditioning}__dec_emb_dim_{dec_emb_dim}__phoneme_cls_{'_'.join(map(str, args['phoneme_cls']))}__seed_{seed}"

                                )

                                main(args)
