from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

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
from text2brain.models.loss import ELBOLoss, GECOLoss
from text2brain.models.vae import VAE, CondVAE, logvar_to_std
from text2brain.visualization import plot_brain_signal_animation, plot_tsne, plot_means_and_stds, plot_elbo_loss, plot_geco_loss, plot_single_image, plot_original_vs_reconstructed_image
from utils import dump_args, load_args, load_pkl, set_seeds


def dicts_match(d1: dict, d2:dict, keys_to_ignore: List[str]):
    """Compare two dictionaries excluding the keys_to_ignore. """
    for key in d1.keys():
        if key not in keys_to_ignore and d1.get(key) != d2.get(key):
            v1 = d1.get(key)
            v2 = d2.get(key)
            if isinstance(v1, tuple) or isinstance(v1, list):  # if one is list, one tuple, 
                if not list(v1) == list(v2):
                    return False
    for key in d2.keys():
        if key not in keys_to_ignore and d1.get(key) != d2.get(key):
            return False
    return True


def find_matching_args_files(base_dir: Path, target_args: dict, keys_to_ignore: List[str]) -> Optional[List[Path]]:
    paths = []
    for file_path in base_dir.rglob("args.json"): 
        if "VAE_unconditional_20240805_150829" in file_path.parts:
            verbose=True
        else:
            verbose=False
        if "bin" in file_path.parts:
            continue
        args = load_args(file_path)
        equal_dicts = dicts_match(args, target_args, keys_to_ignore)
        if equal_dicts:
            paths.append(file_path)

    if len(paths) > 0:
        return paths
    else:
        return None


def load_checkpoint(model, args, base_dir, keys_to_ignore):
    print("LOOKING FOR CHECKPOINT ...")
    if "start_epoch" not in args.keys():
        args["start_epoch"] = 0

    # finda model where the args are same except key_to_ignore
    matching_arg_files = find_matching_args_files(base_dir, args, keys_to_ignore)

    if matching_arg_files is None:
        print("No checkpoint found.")
        return model, args, None

    max_epoch = -1
    best_weights_path = None

    for p in matching_arg_files:
        print(f"p = {p}")

    for arg_path in matching_arg_files:
        path = arg_path.parent

        weights_path = None
        for file_path in path.rglob("*"):
            if "modelWeights_before_passing_max_beta" in file_path.name:
                weights_path = file_path
                break

        if weights_path is not None:
            loaded_stats = load_args(path / "trainingStats.json")
            epoch = loaded_stats["epoch_ckps"]["modelWeights_before_passing_max_beta"]
            if epoch > max_epoch:
                max_epoch = epoch
                best_weights_path = weights_path
            
    if best_weights_path is not None:
        print(f"\nSelected weights_path = {best_weights_path} \n with max_epoch = {max_epoch}")
        best_dir = best_weights_path.parent
        loaded_stats = load_args(best_dir / "trainingStats.json")
        model.load_state_dict(torch.load(best_weights_path))

        args["start_epoch"] = max_epoch
        args["checkpoint_loaded_from"] = str(best_weights_path)

        return model, args, loaded_stats

    print("No modelWeights file found.")
    return model, args, None


def main(args: dict) -> None:
    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
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

    phoneme_cls = args["phoneme_cls"]
    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": phoneme_cls}

    # load train and test data
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

    if args["conditioning"] == None:
        model = VAE(
            latent_dim=args["latent_dim"], 
            input_shape=tuple(args["input_shape"]), 
            classes=phoneme_cls,
            device=device,
            dec_hidden_dim=args["dec_hidden_dim"],
        )
    else:
        model = CondVAE(
            latent_dim=args["latent_dim"], 
            input_shape=tuple(args["input_shape"]), 
            classes=phoneme_cls, 
            device=device, 
            dec_hidden_dim=args["dec_hidden_dim"],
            conditioning=args["conditioning"], 
            n_layers_film=args["n_layers_film"],
            dec_emb_dim=args["dec_emb_dim"],
        )
    args["model_class"] = model.__class__.__name__
    print(f"model = {model}")

    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
    if args["loss"] == "elbo":
        loss_fn = ELBOLoss(
            reduction=args["loss_reduction"],
            beta=args["elbo_beta"]
        )
    elif args["loss"] == "geco":
        loss_fn = GECOLoss(
            goal=args["geco_goal"],
            step_size=args["geco_step_size"],
            beta_init=args["geco_beta_init"],
            beta_max=args["geco_beta_max"],
            reduction=args["loss_reduction"],
            device=device,
        )

    print(f"loss_fn.__class__.__name__ = {loss_fn.__class__.__name__}")

    stats = None
    if args["load_from_checkpoint"]:
        model, args, stats = load_checkpoint(model, args, out_dir.parent, keys_to_ignore=["geco_max_beta", "output_dir", "plot_dir", "n_epochs"])
        
    if stats is None:
        all_mse = {"train": [], "val": []}
        all_kld = {"train": [], "val": []}
        all_epoch_loss = {"train": [], "val": []}
        all_betas = []
        checkpoints = {}
    else:
        all_mse = stats["mse"]
        all_kld = stats["kld"]
        all_epoch_loss = stats["loss"]
        all_betas = stats["beta"]
        checkpoints = stats["epoch_ckps"]
        # also update current beta in geco loss
        if isinstance(loss_fn, GECOLoss):
            loss_fn.beta = all_betas[-1]
    
    # vae_dir = "vae_conditional" if isinstance(model, CondVAE) else "vae"
    plot_dir = args["plot_dir"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot_dir = {plot_dir}")
    args["plot_dir"] = str(plot_dir)

    dump_args(args, out_dir / "args")
    dump_args(args, out_dir / "args.json")
    dump_args(args, plot_dir / "args.json")

    print("\nArguments:")
    for k, v in args.items():
        print(f"\t{k} = {v}")

    n_batches_train = len(train_dl)
    n_batches_val = len(val_dl)

    start_epoch = args["start_epoch"]
    n_epochs = args["n_epochs"]
    convergence = False

    for epoch in range(start_epoch, n_epochs):
        if convergence: 
            break

        epoch_train_loss, epoch_train_mse, epoch_train_kld = 0.0, 0.0, 0.0
        all_mu = []
        all_std = []

        model.train()
        for i, data in enumerate(train_dl):
            X, y, _, _ = map(lambda x: x.to(device), data)

            optimizer.zero_grad()
            if isinstance(model, VAE):
                X_recon, mu, logvar = model(X)
            elif isinstance(model, CondVAE):
                X_recon, mu, logvar = model(X, y)

            if epoch % 10 == 0:
                all_mu.append(mu.detach().cpu().numpy())
                std = logvar_to_std(logvar.detach().cpu())
                all_std.append(std.numpy())

            results = loss_fn(X_recon, X, mu, logvar)
            results.loss.backward()
            optimizer.step()

            epoch_train_loss += results.loss.item()
            epoch_train_mse += results.mse.item()
            epoch_train_kld += results.kld.item()

            if isinstance(loss_fn, GECOLoss):
                all_betas.append(loss_fn.beta.cpu().detach().numpy().tolist())

            # output training stats
            if i % 100 == 0:
                writer.add_scalar("Loss/Train", results.loss.item(), epoch * len(train_dl) + i)
                writer.add_scalar("MSE/Train", results.mse.item(), epoch * len(train_dl) + i)
                writer.add_scalar("KLD/Train", results.kld.item(), epoch * len(train_dl) + i)

            if i % 500 == 0 and epoch % 10 == 0:
                for j in range(10):
                    plot_original_vs_reconstructed_image(
                        X[j][0].cpu().detach().numpy(),
                        X_recon[j][0].cpu().detach().numpy(),
                        plot_dir / f"reconstructed_image_{epoch}_{i}__cls_{y[j]}.png",
                    )
                    if isinstance(model, VAE):
                        generated_image = model.sample(n_samples=1).cpu().detach().numpy()
                    elif isinstance(model, CondVAE):
                        generated_image = model.sample(y[j].view(-1,)).cpu().detach().numpy()

                    plot_single_image(
                        X=generated_image[0][0],
                        out_file=plot_dir / f"generated_image_{epoch}_{i}__cls_{y[j]}.png",
                        title=f"Generated image, phonem class {y[j]}"
                    )

        train_loss = epoch_train_loss / n_batches_train
        all_epoch_loss["train"].append(train_loss)
        all_mse["train"].append(epoch_train_mse / n_batches_train)
        all_kld["train"].append(epoch_train_kld / n_batches_train)

        # if both kld and mse converge, save model weights and stop training AND mse somewhat close to geco goal.
        if epoch > 1:
            if isinstance(loss_fn, GECOLoss):
                if 0 <= all_mse["train"][-1] - all_mse["train"][-2] < 0.000025:
                    if 0 <= loss_fn.goal - all_mse["train"][-1] < 0.001:
                        if 0 >= all_kld["train"][-1] - all_kld["train"][-2] > - 0.0005:
                            convergence = True
                            model.save_state_dict(out_dir / f"modelWeights_epoch_{epoch}")
                            print(f"Convergence after {epoch} epochs ...")
                            break
                        
            elif isinstance(loss_fn, ELBOLoss):
                if abs(all_mse["train"][-1] - all_mse["train"][-2]) < 0.000025:
                    if abs(all_kld["train"][-1] - all_kld["train"][-2]) < 0.001:
                        convergence = True
                        model.save_state_dict(out_dir / f"modelWeights_epoch_{epoch}")
                        print(f"Convergence after {epoch} epochs ...")
                        break

        # compute the mean and logvar
        if epoch % 10 == 0:
            mean_mu = np.mean(np.concatenate(all_mu, axis=0), axis=0)
            mean_std = np.mean(np.concatenate(all_std, axis=0), axis=0)
            plot_means_and_stds(means=mean_mu, stds=mean_std, phoneme=f"3_31__epoch_{epoch}", out_dir=plot_dir)

        # validate on val set at end of epoch
        epoch_val_loss, epoch_val_mse, epoch_val_kld = 0.0, 0.0, 0.0

        model.eval()
        for i, data in enumerate(val_dl):
            X, y, _, _ = map(lambda x: x.to(device), data)

            if isinstance(model, VAE):
                X_recon, mu, logvar = model(X)
            elif isinstance(model, CondVAE):
                X_recon, mu, logvar = model(X, y)

            results = loss_fn(X_recon, X, mu, logvar)

            epoch_val_loss += results.loss.item()
            epoch_val_mse += results.mse.item()
            epoch_val_kld += results.kld.item()

        all_epoch_loss["val"].append(epoch_val_loss / n_batches_val)
        all_mse["val"].append(epoch_val_mse / n_batches_val)
        all_kld["val"].append(epoch_val_kld / n_batches_val)

        writer.add_scalar("Epoch_Loss/Val", epoch_val_loss / n_batches_val, epoch)
        writer.add_scalar("Epoch_MSE/Val", epoch_val_mse / n_batches_val, epoch)
        writer.add_scalar("Epoch_KLD/Val", epoch_val_kld / n_batches_val, epoch)

        print(
            f"[{epoch}/{n_epochs}] train_loss: {train_loss} val_loss: {epoch_val_loss / n_batches_val}"
        )

        # save training statistics
        stats = {"loss": all_epoch_loss, "mse": all_mse, "kld": all_kld, "beta": all_betas, "epoch_ckps": checkpoints}
        dump_args(stats, plot_dir / "trainingStats.json")
        dump_args(stats, out_dir / "trainingStats.json")

        if isinstance(loss_fn, ELBOLoss):
            plot_elbo_loss(
                all_epoch_loss["train"],
                all_mse["train"],
                all_kld["train"],
                out_file=plot_dir / "losses_train.png",
                title="Training Metrics",
            )
            plot_elbo_loss(
                all_epoch_loss["val"],
                all_mse["val"],
                all_kld["val"],
                out_file=plot_dir / "losses_val.png",
                title="Validation Metrics",
            )
        elif isinstance(loss_fn, GECOLoss):
            plot_geco_loss(
                all_epoch_loss["train"],
                all_mse["train"],
                all_kld["train"],
                all_betas,
                out_file=plot_dir / "losses_train.png",
                geco_goal=loss_fn.goal,
                title="Training Metrics",
            )
            plot_geco_loss(
                all_epoch_loss["val"],
                all_mse["val"],
                all_kld["val"],
                all_betas,
                out_file=plot_dir / "losses_val.png",
                geco_goal=loss_fn.goal,
                title="Validation Metrics",
            )

    if not convergence:
        model.save_state_dict(out_dir / f"modelWeights_epoch_{epoch}")
        print(f"Model did not converge, save checkpoint after {epoch} epochs ...")

    writer.close()


def run_latent_dim_experiment(latent_dims: List[int]):
    args = {}
    args["train_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl")
    args["val_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl")

    args["seed"] = 0
    args["device"] = "cuda"
    args["batch_size"] = 64
    args["phoneme_cls"] = [3, 31]
    args["conditioning"] = "film"
    args["n_layers_film"] = 2
    args["load_from_checkpoint"] = False

    args["loss"] = "geco"
    args["loss_reduction"] = "mean"

    args["geco_goal"] = 0.05
    args["geco_beta_init"] = 1e-3
    args["geco_step_size"] = 1e-2
    args["geco_beta_max"] = 10.

    args["input_shape"] = [4, 64, 32]
    args["dec_emb_dim"] = None
    args["dec_hidden_dim"] = 256 

    args["start_epoch"] = 0
    args["n_epochs"] = 300
    args["lr"] = 1e-3

    args["transform"] = "softsign"

    for latent_dim in latent_dims: 
        args["latent_dim"] = latent_dim

        args["output_dir"] = (
            f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment__film_updated__2_layers_2_per_film/VAE__latent_dim_{args['latent_dim']}_cond_{args['conditioning']}"
            # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_concat__phoneme_cls_{args['phoneme_cls']}_dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}"
        )

        args["plot_dir"] = (
            ROOT_DIR
            / "evaluation"
            / "vae_experiments"
            / "run_20240816__VAE_latent_dim_experiment__film_updated__2_layers_2_per_film"  # "run_20240816__VAE_experiment_conditioning" 
            / f"vae__latent_dim_{args['latent_dim']}"  # f"vae__conditioning_concat__phoneme_cls_{args['phoneme_cls']}__dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}" 
        )

        main(args)


def run_conditioning_experiment(latent_dim: int, dec_hidden_dim: int):
    films = [
        ("film", 2, None, [3, 31]),
    ]
    separates = [
        (None, None, None, [3]),
        (None, None, None, [31]),
    ]
    concats = [
        ("concat",  None, 2, [3, 31]),
        ("concat",  None, 4, [3, 31]),
        ("concat",  None, 8, [3, 31]),
        ("concat",  None, 16, [3, 31]),
        ("concat",  None, 32, [3, 31]),
        ("concat",  None, 64, [3, 31]),
        ("concat",  None, 128, [3, 31]),
    ]

    conditions = films + separates + concats

    for cond, n_film, emb_dim, phoneme_cls in conditions:                                                 
        args = {}
        args["train_set_path"] = (
            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
        )
        args["val_set_path"] = (
            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl"
        )
        args["test_set_path"] = (
            "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl"
        )

        args["load_from_checkpoint"] = False

        args["seed"] = 0
        args["device"] = "cuda"
        args["batch_size"] = 64
        args["input_shape"] = [4, 64, 32]
        args["latent_dim"] = latent_dim

        args["loss"] = "geco"
        args["loss_reduction"] = "mean"
        args["geco_goal"] = 0.05
        args["geco_beta_init"] = 1e-3
        args["geco_step_size"] = 1e-2
        args["geco_beta_max"] = 1e-1

        args["phoneme_cls"] = phoneme_cls
        args["conditioning"] = cond
        args["n_layers_film"] = n_film
        
        args["dec_hidden_dim"] = dec_hidden_dim
        args["dec_emb_dim"] = emb_dim

        args["start_epoch"] = 0
        args["n_epochs"] = 300
        args["lr"] = 1e-3

        args["transform"] = "softsign"

        args["output_dir"] = (
            f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning__film_updated/VAE__conditioning_{cond}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}__latent_dim_{args['latent_dim']}__dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}"
        )
        args["plot_dir"] = (
            ROOT_DIR
            / "evaluation"
            / "vae_experiments"  
            / "run_20240816__VAE_experiment_conditioning__film_updated" 
            / f"vae__conditioning_{cond}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}__latent_dim_{args['latent_dim']}__dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}" 
        )

        main(args)


if __name__ == "__main__":
    print("Starting VAE training ...")

    latent_dim_experiment = True
    conditioning_experiment = False
    elbo_vs_geco_experiment = False

    if latent_dim_experiment:
        run_latent_dim_experiment(latent_dims=[256, ])

    if conditioning_experiment:
        for hid_dim in [256,]:  # [ (512, 128), (512, 256), (512, 512)]
            for lat_dim in [512, ]:
                run_conditioning_experiment(
                    latent_dim=lat_dim, 
                    dec_hidden_dim=hid_dim
                )

    if elbo_vs_geco_experiment:
        args = {}
        args["load_from_checkpoint"] = False

        args["n_layers_film"] = 2
        args["seed"] = 0
        args["device"] = "cuda"
        args["batch_size"] = 64
        args["phoneme_cls"] = [3, 31]
        args["conditioning"] = "film"
        args["dec_hidden_dim"] = 256
        args["dec_emb_dim"] = None
        args["input_shape"] = [4, 64, 32]

        args["start_epoch"] = 0
        args["n_epochs"] = 300
        args["lr"] = 1e-3

        args["transform"] = "softsign"

        for latent_dim in [128, 256, 512]:
            for loss in ["elbo", "geco"]:
                for beta_init in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]: 
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")

                    args["latent_dim"] = latent_dim

                    args["loss"] = loss
                    args["loss_reduction"] = "mean"
                    if loss == "elbo":
                        args["elbo_beta"] = beta_init
                    if loss == "geco":
                        args["geco_goal"] = 0.05
                        args["geco_beta_init"] = beta_init
                        args["geco_step_size"] = 1e-2
                        args["geco_beta_max"] = 1.0

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
                        f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_elbo_vs_geco/VAE__latent_dim_{latent_dim}__loss_{loss}__beta_init_{beta_init}"
                    )
                    args["plot_dir"] = (
                        ROOT_DIR
                        / "evaluation"
                        / "vae_experiments"
                        / "run_20240816__VAE_experiment_elbo_vs_geco__film_updated"
                        / f"vae__latent_dim_{latent_dim}__loss_{loss}__beta_init_{beta_init}"
                    )

                    main(args)


    # # unconditional VAE, trained on two classes (3, 31)
    # args = {}
    # args["train_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl")
    # args["val_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl")
    # args["test_set_path"] = ("/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl")

    # args["seed"] = 0
    # args["device"] = "cuda"

    # for lr in [1e-3, 1e-4]:  # [1e-3, 1e-4, 1e-5]:
    #     now = datetime.now()
    #     timestamp = now.strftime("%Y%m%d_%H%M%S")

    #     args["load_from_checkpoint"] = False

    #     args["batch_size"] = 64
    #     args["phoneme_cls"] = [3, 31]
    #     args["conditioning"] = None

    #     args["loss"] = "geco"
    #     args["loss_reduction"] = "mean"

    #     args["geco_goal"] = 0.05
    #     args["geco_beta_init"] = 1e-3
    #     args["geco_step_size"] = 1e-2
    #     args["geco_beta_max"] = 1e-1

    #     args["input_shape"] = [4, 64, 32]
    #     args["latent_dim"] = 256
    #     args["dec_hidden_dim"] = 256

    #     args["start_epoch"] = 0
    #     args["n_epochs"] = 200
    #     args["lr"] = lr

    #     args["transform"] = "softsign"

    #     # vae_dir_name = "VAE_unconditional" if len(args["phoneme_cls"]) > 1 else "VAE_conditional"
    #     args["output_dir"] = (
    #         f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_unconditional_classes_3_31/VAE__lr_{args['lr']}"
    #     )
    #     args["plot_dir"] = (
    #         ROOT_DIR
    #         / "evaluation"
    #         / "vae_experiments"
    #         / "run_20240816__VAE_unconditional_classes_3_31"
    #         / f"vae__lr_{args['lr']}"
    #         # / f"reconstructed_images_model_input_shape_{'_'.join(map(str, args['input_shape']))}__cond_{args['conditioning']}__dec_emb_dim_{args['dec_emb_dim']}__latent_dim_{args['latent_dim']}__loss_{args['loss']}_{args['geco_goal']}_{args['geco_step_size']}_{args['geco_beta_init']}_{args['geco_beta_max']}__lr_{args['lr']}__bs_{batch_size}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}__dec_hid_dim_{args['dec_hidden_dim']}"
    #     )

    #     main(args)


