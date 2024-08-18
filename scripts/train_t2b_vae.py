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
from text2brain.models.vae_hierarchical import HierarchicalCondVAE
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
                    kernel_size=args["gaussian_smoothing_kernel_size"],
                    sigma=args["gaussian_smoothing_sigma"],
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

    if args["hierarchical"]:
        model = HierarchicalCondVAE(
            latent_dims=args["hierarchical_latent_dims"],
            input_shape=tuple(args["input_shape"]),
            classes=phoneme_cls,
            device=device,
            dec_hidden_dim=args["dec_hidden_dim"],
            conditioning=args["conditioning"],
            n_layers_film=args["n_layers_film"],
            # dec_emb_dim=dec_emb_dim,
        )
    elif len(phoneme_cls) == 1:
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
            # dec_emb_dim=args["dec_emb_dim"],
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
    
    first_ckp = True
    second_ckp = True

    for epoch in range(start_epoch, n_epochs):
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
            elif isinstance(model, HierarchicalCondVAE):
                X_recon, mus, logvar = model(X, y)
                print(f"len(mus) = {len(mus)}")
                print(f"len(mus) = {len(mus)}")
                for mu in mus:
                    print(f"mu.size() = {mu.size()}")
                for logvar in logvar:
                    print(f"logvar.size() = {logvar.size()}")
                

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

            if i % 500 == 0 and (epoch % 10 == 0 or (epoch < 20 and epoch % 2 == 0)):
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

        # compute the mean and logvar
        if epoch % 10 == 0:
            mean_mu = np.mean(np.concatenate(all_mu, axis=0), axis=0)
            mean_std = np.mean(np.concatenate(all_std, axis=0), axis=0)
            plot_means_and_stds(means=mean_mu, stds=mean_std, phoneme=f"3_31__epoch_{epoch}", out_dir=plot_dir)

            # plot_tsne(all_mu, title="t-SNE visualization of latent space", out_file=plot_dir / "tsne_mu.png")
            # plot_tsne(all_std, title="t-SNE visualization of latent space", out_file=plot_dir / "tsne_std.png")

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
        if epoch >= 90 and epoch % 10 == 0:
            model.save_state_dict(out_dir / f"modelWeights_epoch_{epoch}")
        
        # # save checkpoints
        # if isinstance(loss_fn, GECOLoss):
        #     if first_ckp:
        #         if train_loss > loss_fn.goal:
        #             model.save_state_dict(out_dir / f"modelWeights_before_passing_goal_{loss_fn.goal}.pt")
        #             checkpoints[f"modelWeights_before_passing_goal"] = epoch
        #         else:
        #             first_ckp = False

        #     if second_ckp:
        #         if loss_fn.beta < loss_fn.beta_max:
        #             model.save_state_dict(out_dir / f"modelWeights_before_passing_max_beta_{args['geco_beta_max']}.pt")
        #             checkpoints[f"modelWeights_before_passing_max_beta"] = epoch
        #         else:
        #             second_ckp = False

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

    writer.close()


if __name__ == "__main__":
    print("Starting VAE training ...")

    for input_shape in [[4, 64, 32],]:  # [[4, 64, 32], [1, 256, 32], [128, 8, 8]]:
        for loss in ["geco"]:  # ["elbo", "geco"]:
            for lr in [1e-3]:  # [1e-3, 1e-4, 1e-5]:
                for n_layers_film in [None]:  #, 1]:
                    for geco_goal in [0.05]:  # , 0.037]:
                        for geco_step_size in [1e-2]:  # , 1e-3, 1e-4]:
                            for beta_init in [1e-3]:  # 1e-3]:  # , 1e-5, 1e-7]:
                                for dec_emb_dim in [16, 32, 64, 128, 8]:
                                    for geco_beta_max in [1e-1]:
                                        for conditioning in ["concat"]:
                                            for latent_dim in [256]:  # , 128, 100]:
                                                for dec_hidden_dim in [256, 512]:
                                                    for phoneme_cls in [[3, 31],]:
                                                        if ((dec_emb_dim == 8 or dec_emb_dim == 64) and dec_hidden_dim == 512)  or dec_hidden_dim == 128:
                                                                

                                                            args = {}

                                                            args["hierarchical"] = False
                                                            # args["hierarchical_latent_dims"] = [64, 32, 16]

                                                            now = datetime.now()
                                                            timestamp = now.strftime("%Y%m%d_%H%M%S")

                                                            args["load_from_checkpoint"] = False

                                                            args["n_layers_film"] = n_layers_film

                                                            args["seed"] = 0
                                                            args["device"] = "cuda"
                                                            args["batch_size"] = 64
                                                            args["phoneme_cls"] = [phoneme_cls] if isinstance(phoneme_cls, int) else phoneme_cls  # list(range(1, 40))
                                                            args["conditioning"] = conditioning

                                                            args["loss"] = loss
                                                            args["loss_reduction"] = "mean"
                                                            if loss == "elbo":
                                                                args["elbo_beta"] = beta_init
                                                            if loss == "geco":
                                                                args["geco_goal"] = geco_goal
                                                                args["geco_beta_init"] = beta_init
                                                                args["geco_step_size"] = geco_step_size
                                                                args["geco_beta_max"] = geco_beta_max

                                                            args["input_shape"] = input_shape
                                                            args["latent_dim"] = latent_dim
                                                            args["dec_hidden_dim"] = dec_hidden_dim
                                                            args["dec_emb_dim"] = dec_emb_dim

                                                            args["start_epoch"] = 0
                                                            args["n_epochs"] = 150
                                                            args["lr"] = lr

                                                            args["transform"] = "softsign"
                                                            args["gaussian_smoothing_kernel_size"] = 20
                                                            args["gaussian_smoothing_sigma"] = 2.0

                                                            args["train_set_path"] = (
                                                                "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
                                                            )
                                                            args["val_set_path"] = (
                                                                "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl"
                                                            )
                                                            args["test_set_path"] = (
                                                                "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl"
                                                            )
                                                            # vae_dir_name = "VAE_unconditional" if len(args["phoneme_cls"]) > 1 else "VAE_conditional"
                                                            # args["output_dir"] = (
                                                            #     f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_hierarchical/vae__hierarchical_latent_dims_{'_'.join(map(str, args['hierarchical_latent_dims']))}"
                                                            # )
                                                            args["output_dir"] = (
                                                                f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_concat__phoneme_cls_3_31__dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}"
                                                            )
                                                            args["plot_dir"] = (
                                                                ROOT_DIR
                                                                / "evaluation"
                                                                / "vae_experiments"  # "vae_hierarchical"
                                                                / "run_20240816__VAE_experiment_conditioning"  # "run_20240817__hierarchical_vae"
                                                                / f"vae__conditioning_concat__phoneme_cls_3_31__dec_emb_dim_{args['dec_emb_dim']}__dec_hidden_dim_{args['dec_hidden_dim']}"  # f"vae__hierarchical_latent_dims_{'_'.join(map(str, args['hierarchical_latent_dims']))}"
                                                            )

                                                            main(args)




    # for input_shape in [[4, 64, 32],]:  # [[4, 64, 32], [1, 256, 32], [128, 8, 8]]:
    #     for loss in ["elbo", "geco"]:  # ["elbo", "geco"]:
    #         for lr in [1e-3]:  # [1e-3, 1e-4, 1e-5]:
    #             for n_layers_film in [2]:  #, 1]:
    #                 for geco_goal in [0.05]:  # , 0.037]:
    #                     for geco_step_size in [1e-2]:  # , 1e-3, 1e-4]:
    #                         for beta_init in [1.0, 1e-1, 1e-2, 1e-3, 1e-3]:  # 1e-3]:  # , 1e-5, 1e-7]:
    #                             for dec_emb_dim in [None]:  # [8, 16, 32]:
    #                                 for geco_beta_max in [1e-1]:
    #                                     for conditioning in ["film"]:
    #                                         for latent_dim in [256]:  # , 128, 100]:
    #                                             for dec_hidden_dim in [256]:
    #                                                 for phoneme_cls in [[3, 31],]:

    #                                                     now = datetime.now()
    #                                                     timestamp = now.strftime("%Y%m%d_%H%M%S")

    #                                                     args = {}
    #                                                     args["load_from_checkpoint"] = False

    #                                                     args["n_layers_film"] = n_layers_film

    #                                                     args["seed"] = 0
    #                                                     args["device"] = "cuda"
    #                                                     args["batch_size"] = 64
    #                                                     args["phoneme_cls"] = [phoneme_cls] if isinstance(phoneme_cls, int) else phoneme_cls  # list(range(1, 40))
    #                                                     args["conditioning"] = conditioning

    #                                                     args["loss"] = loss
    #                                                     args["loss_reduction"] = "mean"
    #                                                     if loss == "elbo":
    #                                                         args["elbo_beta"] = beta_init
    #                                                     if loss == "geco":
    #                                                         args["geco_goal"] = geco_goal
    #                                                         args["geco_beta_init"] = beta_init
    #                                                         args["geco_step_size"] = geco_step_size
    #                                                         args["geco_beta_max"] = geco_beta_max

    #                                                     args["input_shape"] = input_shape
    #                                                     args["latent_dim"] = latent_dim
    #                                                     args["dec_hidden_dim"] = dec_hidden_dim
    #                                                     args["dec_emb_dim"] = dec_emb_dim

    #                                                     args["start_epoch"] = 0
    #                                                     args["n_epochs"] = 150
    #                                                     args["lr"] = lr

    #                                                     args["transform"] = "softsign"
    #                                                     args["gaussian_smoothing_kernel_size"] = 20
    #                                                     args["gaussian_smoothing_sigma"] = 2.0

    #                                                     args["train_set_path"] = (
    #                                                         "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    #                                                     )
    #                                                     args["val_set_path"] = (
    #                                                         "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_VAL_SPLIT.pkl"
    #                                                     )
    #                                                     args["test_set_path"] = (
    #                                                         "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits_TEST_SPLIT.pkl"
    #                                                     )
    #                                                     # vae_dir_name = "VAE_unconditional" if len(args["phoneme_cls"]) > 1 else "VAE_conditional"
    #                                                     args["output_dir"] = (
    #                                                         f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_elbo_vs_geco/VAE__loss_{args['loss']}__beta_init_{args['elbo_beta']}"
    #                                                     )
    #                                                     args["plot_dir"] = (
    #                                                         ROOT_DIR
    #                                                         / "evaluation"
    #                                                         / "vae_experiments"
    #                                                         / "run_20240816__VAE_experiment_elbo_vs_geco"
    #                                                         / f"vae__loss_{args['loss']}__beta_init_{args['elbo_beta']}"
    #                                                         # / f"reconstructed_images_model_input_shape_{'_'.join(map(str, args['input_shape']))}__cond_{args['conditioning']}__dec_emb_dim_{args['dec_emb_dim']}__latent_dim_{args['latent_dim']}__loss_{args['loss']}_{args['geco_goal']}_{args['geco_step_size']}_{args['geco_beta_init']}_{args['geco_beta_max']}__lr_{args['lr']}__gs_{args['gaussian_smoothing_kernel_size']}_{args['gaussian_smoothing_sigma']}__bs_{batch_size}__phoneme_cls_{'_'.join(map(str, phoneme_cls))}__dec_hid_dim_{args['dec_hidden_dim']}"
    #                                                     )

    #                                                     main(args)
