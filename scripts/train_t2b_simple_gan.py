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
        lr_g=args["lr_g"],
        lr_d=args["lr_d"],
        pool=args["pool"],
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
    print(f"gan = {gan}")

    plot_dir = Path(args["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot_dir = {plot_dir}")

    dump_args(args, out_dir / "args")
    dump_args(args, out_dir / "args.json")
    dump_args(args, plot_dir / "args.json")

    G_losses = []
    D_losses = []
    
    noise = torch.randn(3, gan._g.latent_dim, device=gan.device)

    n_epochs = args["n_epochs"]
    n_steps = int(len(train_dl) / (n_critic + 1))

    print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")
    print(f"len(train_dl) = {len(train_dl)}")
    print(f"n_steps = {n_steps}")

    for epoch in range(n_epochs):
        G_losses_epoch = []
        D_losses_epoch = []

        print(f"epoch = {epoch}")
        for i in range(n_steps):
            if (i % 20 == 0 and i < 100 and epoch == 0) or (i == 0 and epoch % 5 == 0):
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

                    # plot_single_image(
                    #     X=x[0][0],
                    #     out_file=plot_dir / f"real_image_{epoch}_{i}__cls_{label}.png",
                    #     title=f"Real image \n(phoneme cls {label})"
                    # )

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
        gan.save_state_dict(file)
        # torch.save(gan.state_dict(), file)

        plot_gan_losses(
            G_losses,
            D_losses,
            out_file=plot_dir / f"gan_losses_nclasses_{len(phoneme_cls)}.png",
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
    for lr_g in [0.00001]:  #, 1e-5, 1e-3]:  # [1e-3, 1e-4, 1e-5]:
        for lr_d in [0.00001]:  # , 0.05, 0.5]:
            for n_critic in [5]:  # , 2]:
                for pool in ["none"]:
                    for loss in ["wasserstein", ]:
                        for latent_dim in [256]:
                            # if (factor_lr_d == 0.05 and n_critic==2) or (factor_lr_d == 0.01 and n_critic==1):

                            now = datetime.now()
                            timestamp = now.strftime("%Y%m%d_%H%M%S")

                            print("in main ...")
                            args = {}
                            args["seed"] = 0
                            args["device"] = "cuda"
                            args["timestamp"] = timestamp

                            args["phoneme_cls"] = [3]  # [3, 31]

                            args["batch_size"] = 128

                            args["latent_dim"] = latent_dim
                            args["input_shape"] = (4, 64, 32)

                            args["n_epochs"] = 10_000
                            args["loss"] = loss
                            args["lr_g"] = lr_g
                            args["lr_d"] = lr_d
                            # args["lr"] = 0.0001
                            args["n_critic"] = n_critic
                            args["transform"] = "softsign"
                            args["gaussian_smoothing_kernel_size"] = 20
                            args["gaussian_smoothing_sigma"] = 2.0
                            args["pool"] = pool

                            # args["vae_path"] = "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120"
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
                                f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_{timestamp}__phoneme_cls_{'_'.join(map(str, args['phoneme_cls']))}"
                            )
                            args["plot_dir"] = str((
                                ROOT_DIR
                                / "evaluation"
                                / "gan"
                                / "test__run_20240817_gradient_penatly"
                                / f"gan_{args['timestamp']}_in_{'_'.join(map(str, args['input_shape']))}__latdim_{args['latent_dim']}__lrg_{args['lr_g']}__lrd_{args['lr_d']}__phoneme_cls_{'_'.join(map(str, args['phoneme_cls']))}"
                            ))

                            main(args)
