import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data.dataset import PhonemeDataset
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
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models.model_interface_load import load_t2b_gen_model
from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.visualization import plot_brain_signal_animation, plot_single_image
from utils import dump_args, load_args, load_pkl, set_seeds



def main(args: dict) -> None:
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
                    kernel_size=args["gaussian_smoothing_kernel_size"],
                    sigma=args["gaussian_smoothing_sigma"],
                    dim=1,
                ),
                SoftsignTransform(),
            ]
        )

    phoneme_cls = args["phoneme_cls"]
    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": phoneme_cls}

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

    gan = PhonemeImageGAN(
        input_shape=args["input_shape"],
        latent_dim=args["latent_dim"],
        phoneme_cls=args["phoneme_cls"],
        device=device,
        n_critic=args["n_critic"],
        clip_value=args["clip_value"],
        lr_g=args["lr_g"],
        lr_d=args["lr_d"],
        pool=args["pool"],
        smooth_labels=args["smooth_labels"],
        noisy_labels=args["noisy_labels"],
        loss=args["loss"],
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

    plot_dir = (
        ROOT_DIR
        / "evaluation"
        / "gan"
        / "run_20240812_0"
        / f"gan_{args['timestamp']}_in_{'_'.join(map(str, args['input_shape']))}__latdim_{args['latent_dim']}__lrg_{args['lr_g']}__lrd_{args['lr_d']}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot_dir = {plot_dir}")
    args["plot_dir"] = str(plot_dir)

    dump_args(args, out_dir / "args")
    dump_args(args, out_dir / "args.json")
    dump_args(args, plot_dir / "args.json")

    G_losses = []
    D_losses = []
    D_accs = []
    
    noise = torch.randn(3, gan._g.latent_dim, device=gan.device)

    n_epochs = args["n_epochs"]

    for epoch in range(n_epochs):
        print(f"epoch = {epoch}")
        for i, data in enumerate(train_dl):
            x, y, _, _ = data
            if (i % 10 == 0 and i < 100 and epoch == 0) or (i == 0):
                # plot a single image and save to evaluation folder
                for label in phoneme_cls:
                    for j in range(noise.size(0)):
                        label_t = torch.tensor([label])

                        # generated_image = gan.generate(label=label_t).cpu().detach()
                        generated_image = gan.generate_from_given_noise(noise=noise[j].unsqueeze(0), label=label_t).cpu().detach()
                        
                        plot_single_image(
                            X=generated_image[0][0],
                            out_file=plot_dir / f"generated_image_{epoch}_{i}_{j}__cls_{label}.png",
                            title=f"GAN generated image \n(phoneme cls {label}, epoch {epoch})"
                        )

                    plot_single_image(
                        X=x[0][0],
                        out_file=plot_dir / f"real_image_{epoch}_{i}__cls_{label}.png",
                        title=f"Real image \n(phoneme cls {label})"
                    )

            loss_D, loss_G, acc_D = gan.train_step(x, y)

            # save losses for plotting
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            D_accs.append(acc_D.item())

            # output training stats
            if i % 250 == 0:
                print(
                    f"[{epoch}/{n_epochs}][{i}/{len(train_dl)}] Loss_D: {loss_D.item()} Loss_G: {loss_G.item()}"
                )

        # save GAN
        file = out_dir / f"modelWeights"
        print(f"Storing GAN weights to: {file}")
        gan.save_state_dict(file)
        # torch.save(gan.state_dict(), file)

        plot_gan_losses(
            G_losses,
            D_losses,
            out_file=plot_dir / f"gan_losses_nclasses_{len(phoneme_cls)}.png",
        )

        plot_accuracies(
            D_accs,
            out_file=plot_dir / f"discriminator_accuracy_train.png",
            title=f"Discriminator accuracies during training ({len(train_dl)} iterations per epoch)"
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
    for lr_g in [1e-3]:  #, 1e-5, 1e-3]:  # [1e-3, 1e-4, 1e-5]:
        for factor_lr_d in [1, 0.5, 0.2]:  # , 0.5]:
            for n_critic in [1, 2]:  # , 2]:
                for pool in ["avg"]:
                    for smooth_labels in [0.1]:
                        for noisy_labels in [0.05, 0.01]: # TODO
                            for loss in ["bce with flipped labels"]:  # , "wasserstein", ]:

                                now = datetime.now()
                                timestamp = now.strftime("%Y%m%d_%H%M%S")

                                print("in main ...")
                                args = {}
                                args["seed"] = 0
                                args["device"] = "cuda"
                                args["timestamp"] = timestamp

                                args["phoneme_cls"] = [3]  # [3, 31]

                                args["batch_size"] = 64

                                args["latent_dim"] = 256
                                args["input_shape"] = (4, 64, 32)
                                # args["input_shape"] = (128, 8, 8)

                                # args["ngf"] = None  # 64
                                # args["ndf"] = None  # 64

                                args["n_epochs"] = 40
                                args["loss"] = loss
                                args["lr_g"] = lr_g
                                args["lr_d"] = lr_g * factor_lr_d
                                # args["lr"] = 0.0001
                                args["clip_value"] = 0.01
                                args["n_critic"] = n_critic
                                args["transform"] = "softsign"
                                args["gaussian_smoothing_kernel_size"] = 20
                                args["gaussian_smoothing_sigma"] = 2.0
                                args["pool"] = pool
                                args["smooth_labels"] = smooth_labels
                                args["noisy_labels"] = noisy_labels

                                args["vae_path"] = "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120"
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
                                    f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/PhonemeImageGAN_{timestamp}__phoneme_cls_{'_'.join(map(str, args['phoneme_cls']))}"
                                )

                                main(args)
