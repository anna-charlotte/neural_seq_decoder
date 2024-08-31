import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.augmentations import GaussianSmoothing
from data.dataset import PhonemeDataset, SyntheticPhonemeDataset
from neural_decoder.model_phoneme_classifier import train_phoneme_classifier
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from text2brain.models.model_interface_load import load_t2b_gen_model
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from text2brain.visualization import plot_aurocs_with_error_bars
from utils import load_pkl, set_seeds
from evaluation import compute_auroc_with_stderr, compute_auroc_with_confidence_interval


def plot_vae_decoder_weights(vae, out_file: Path, title: str = 'Fully Connected Layer Weights') -> None:
    fc_weights = vae.decoder.fc[0].weight.data.detach().cpu().numpy()

    conv_layers = [layer for layer in vae.decoder.model if isinstance(layer, nn.ConvTranspose2d)]
    conv_weights = [layer.weight.data.cpu().numpy() for layer in conv_layers]

    # Plotting the heatmaps of the weights
    plt.figure(figsize=(15, 5))

    # Fully connected layer heatmap
    plt.subplot(1, len(conv_weights) + 1, 1)

    sns.heatmap(fc_weights, cmap='viridis')
    plt.title(title)
    fc_mean = np.mean(fc_weights[:, :256])
    fc_max = np.amax(fc_weights[:, :256])
    fc_min = np.amin(fc_weights[:, :256])
    print(f"fc_mean = {fc_mean}")
    print(f"fc_max = {fc_max}")
    print(f"fc_min = {fc_min}")

    y_dec_mean = np.mean(fc_weights[:, 256:])
    y_dec_max = np.amax(fc_weights[:, 256:])
    y_dec_min = np.amin(fc_weights[:, 256:])
    print(f"y_dec_mean = {y_dec_mean}")
    print(f"y_dec_max = {y_dec_max}")
    print(f"y_dec_min = {y_dec_min}")

    # Convolutional layers heatmaps
    for i, weight_matrix in enumerate(conv_weights):
        plt.subplot(1, len(conv_weights) + 1, i + 2)
        sns.heatmap(weight_matrix.reshape(weight_matrix.shape[0], -1), cmap='viridis')
        plt.title(f'ConvTranspose2d Layer {i + 1} Weights')

    plt.tight_layout()
    plt.savefig(out_file)


def get_label_distribution(dataset):
    label_counts = Counter()

    for sample in dataset:
        labels = sample[1].item()
        label_counts.update([labels])

    return label_counts




def main(args: dict) -> None:
    print("Training phoneme classifier with the following arguments:")
    for k, v in args.items():
        print(f"{k}: {v}")

    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args["batch_size"]
    device = args["device"]

    phoneme_ds_filter = {"correctness_value": args["correctness_value"], "phoneme_cls": args["phoneme_cls"]}
    args["phoneme_ds_filter"] = phoneme_ds_filter
    phoneme_classes = args["phoneme_cls"]

    transform = None
    if args["transform"] == "softsign":
        transform = transforms.Compose(
            [
                TransposeTransform(0, 1),
                ReorderChannelTransform(),
                AddOneDimensionTransform(dim=0),
                GaussianSmoothing(
                    256,
                    kernel_size=20.0,
                    sigma=2.0,
                    dim=1,
                ),
                SoftsignTransform(),
            ]
        )
    
    # load generative models
    if isinstance(args["generative_model_weights_path"], list):
        gen_models = []
        for weights_path in args["generative_model_weights_path"]:
            model = load_t2b_gen_model(weights_path=weights_path)
            gen_models.append(model)
    else:
        gen_models = [load_t2b_gen_model(weights_path=args["generative_model_weights_path"])]

    # load real val set
    val_dl_real = get_data_loader(
        data=load_pkl(args["val_set_path"]),
        batch_size=1,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=None,
        transform=transform,
    )
    val_dl_real.name = "val-real"

    # load real test set
    test_dl_real = get_data_loader(
        data=load_pkl(args["test_set_path"]),
        batch_size=1,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=None,
        transform=transform,
    )
    test_dl_real.name = "test-real"

    output = train_phoneme_classifier(
        gen_models=gen_models,
        n_samples_train_syn=args["generative_model_n_samples_train"],
        n_samples_val=args["generative_model_n_samples_val"],
        val_dl_real=val_dl_real, 
        test_dl_real=test_dl_real,
        phoneme_classes=phoneme_classes,
        input_shape=args["input_shape"],
        batch_size=args["batch_size"],
        n_epochs=args["n_epochs"], 
        patience=args["patience"],
        lr=args["lr"],
        device=device,
        out_dir=out_dir,
    )
    

    return output



def evaluate_latent_dim_experiment(
        model_dir: Path, 
        latent_dims: list, 
        n_seeds: int = 5, 
        bootstrap_iters: int = 1_000
    ) -> None:

    args = {}
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    args["n_epochs"] = 100

    args["input_shape"] = (128, 8, 8)
    args["lr"] = 1e-4
    args["batch_size"] = 64
    args["transform"] = "softsign"
    args["patience"] = 20
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    aurocs = []
    errs = []
    _aurocs = []
    _stds = []
    _sems = []
    x_labels = []

    for latent_dim in latent_dims:
        # HANDLE NON CASE
        sub_dir = model_dir / f"VAE__latent_dim_{latent_dim}_cond_film"
        matching_files = list(sub_dir.glob("modelWeights*"))
        assert len(matching_files) == 1, "There are multiple modelWeights files in the given directory."

        weights_file = matching_files[0]
        print(f"weights_file = {weights_file}")

        all_y_true = []
        all_y_pred = []
        seed_aurocs = []

        for seed in range(n_seeds):
            args["seed"] = seed
            args["output_dir"] = (
                f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
            )
            
            args["generative_model_weights_path"] = str(weights_file)
            args["generative_model_n_samples_train"] = 10_000
            args["generative_model_n_samples_val"] = 2_000
            args["generative_model_n_samples_test"] = 2_000

            output = main(args)
            print(f"output['test_acc'] = {output['test_acc']}")
            print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
            print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
            
            all_y_true.append(output["test_y_true"])
            all_y_pred.append(output["test_y_pred"])
            
            auroc, _ = compute_auroc_with_stderr(
                y_true=output["test_y_true"], 
                y_pred=output["test_y_pred"], 
                n_iters=bootstrap_iters, 
            )
            seed_aurocs.append(auroc)

        _auroc = np.mean(seed_aurocs)
        
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=n_seeds, replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        _std = np.std(bootstrapped_aurocs)
        _sem = _std / np.sqrt(bootstrap_iters)

        _aurocs.append(_auroc)
        _stds.append(_std)
        _sems.append(_sem)

        y_true_combined = np.concatenate(all_y_true)
        y_pred_combined = np.concatenate(all_y_pred)    

        auroc, err = compute_auroc_with_stderr(
            y_true=y_true_combined, 
            y_pred=y_pred_combined, 
            n_iters=10
        )

        aurocs.append(auroc)
        errs.append(err)
        x_labels.append(latent_dim)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    plot_aurocs_with_error_bars(
        aurocs=aurocs, 
        errs=errs, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_latent_dim_experiment" / f"{timestamp}_aurocs_with_err__seed_{args['seed']}_2.png",
        xlabel="latent dim",
        title="AUROC performance of VAEs with varying latent dimensions",
    )

    plot_aurocs_with_error_bars(
        aurocs=_aurocs, 
        errs=_stds, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_latent_dim_experiment" / f"{timestamp}_STD_aurocs_with_err__seed_{args['seed']}_2.png",
        xlabel="latent dim",
        title="AUROC performance of VAEs with varying latent dimensions",
    )

    plot_aurocs_with_error_bars(
        aurocs=_aurocs, 
        errs=_sems, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_latent_dim_experiment" / f"{timestamp}_SEM_aurocs_with_err__seed_{args['seed']}_2.png",
        xlabel="latent dim",
        title="AUROC performance of VAEs with varying latent dimensions",
    )


def evaluate_conditioning_experiment(
        model_dir: Path, 
        latent_dim: int, 
        dec_hidden_dim: int, 
        n_seeds: int = 10
    ) -> None:
    args = {}
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    args["n_epochs"] = 100

    args["input_shape"] = (128, 8, 8)
    args["lr"] = 1e-4
    args["batch_size"] = 64
    args["transform"] = "softsign"
    args["patience"] = 20
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    aurocs = []
    errs = []
    _aurocs = []
    _stds = []
    _sems = []
    x_labels = []

    name2dir = {
        "Separate Models": [
            f"VAE__conditioning_None__phoneme_cls_3__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}",
            f"VAE__conditioning_None__phoneme_cls_31__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}"
        ],
        "FiLM": f"VAE__conditioning_film__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}",

        "Concat\n(emb_dim=8)": f"VAE__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_8__dec_hidden_dim_{dec_hidden_dim}",
        "Concat\n(emb_dim=16)": f"VAE__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_16__dec_hidden_dim_{dec_hidden_dim}",
        "Concat\n(emb_dim=32)": f"VAE__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_32__dec_hidden_dim_{dec_hidden_dim}",
        "Concat\n(emb_dim=64)": f"VAE__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_64__dec_hidden_dim_{dec_hidden_dim}",
        "Concat\n(emb_dim=128)": f"VAE__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_128__dec_hidden_dim_{dec_hidden_dim}",
    }
    for model_name, weight_dir in name2dir.items():
        if isinstance(weight_dir, str):
            matching_files = list((model_dir / weight_dir).glob("modelWeights*"))
            assert len(matching_files) == 1, f"There are multiple modelWeights files in the given directory, expected one: {matching_files}"
            weights_file = str(matching_files[0])

        elif isinstance(weight_dir, list):
            weights_file = []
            for dir in weight_dir:
                matching_files = list((model_dir / dir).glob("modelWeights*"))
                assert len(matching_files) == 1, f"There are {len(matching_files)} modelWeights files in the given directory, expected one: {matching_files}"
                weights_file.append(str(matching_files[0]))

        print(f"weights_file = {weights_file}")

        all_y_true = []
        all_y_pred = []
        seed_aurocs = []

        for seed in range(n_seeds):
            args["seed"] = seed
            args["output_dir"] = (
                f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
            )
            
            args["generative_model_weights_path"] = weights_file
            args["generative_model_n_samples_train"] = 10_000
            args["generative_model_n_samples_val"] = 2_000
            args["generative_model_n_samples_test"] = 2_000

            output = main(args)
            print(f"output['test_acc'] = {output['test_acc']}")
            print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
            print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
            
            all_y_true.append(output["test_y_true"])
            all_y_pred.append(output["test_y_pred"])
            
            auroc, _ = compute_auroc_with_stderr(
                y_true=output["test_y_true"], 
                y_pred=output["test_y_pred"], 
                n_iters=1_000, 
            )
            seed_aurocs.append(auroc)

        _auroc = np.mean(seed_aurocs)
        
        bootstrapped_aurocs = []
        for _ in range(1_000):
            sample = np.random.choice(seed_aurocs, size=n_seeds, replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        _std = np.std(bootstrapped_aurocs)
        _sem = _std / np.sqrt(1_000)

        _aurocs.append(_auroc)
        _stds.append(_std)
        _sems.append(_sem)

        y_true_combined = np.concatenate(all_y_true)
        y_pred_combined = np.concatenate(all_y_pred)    

        auroc, err = compute_auroc_with_stderr(
            y_true=y_true_combined, 
            y_pred=y_pred_combined, 
            n_iters=10
        )

        aurocs.append(auroc)
        errs.append(err)
        x_labels.append(model_name)


    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    plot_aurocs_with_error_bars(
        aurocs=aurocs, 
        errs=errs, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_experiment_conditioning" / f"{timestamp}_aurocs_with_err__seed_{args['seed']}__lat_dim_{latent_dim}__dec_hidden_dim_{dec_hidden_dim}__dir_2.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms",
    )

    plot_aurocs_with_error_bars(
        aurocs=_aurocs, 
        errs=_stds, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_experiment_conditioning" / f"{timestamp}_STD_aurocs_with_err__seed_{args['seed']}__lat_dim_{latent_dim}__dec_hidden_dim_{dec_hidden_dim}__dir_2.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms",
    )

    plot_aurocs_with_error_bars(
        aurocs=_aurocs, 
        errs=_sems, 
        x_labels=x_labels, 
        out_file=ROOT_DIR / "evaluation" / "vae_experiments" / "run_20240816__VAE_experiment_conditioning" / f"{timestamp}_SEM_aurocs_with_err__seed_{args['seed']}__lat_dim_{latent_dim}__dec_hidden_dim_{dec_hidden_dim}__dir_2.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms",
    )


if __name__ == "__main__":
    print("in main ...")

    latent_dim_experiment = False
    conditioning_experiment = False
    elbo_vs_geco_experiment = False

    if latent_dim_experiment:
        print("Run latent dim experiment ...")
        model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment_2")
        # model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment__film_updated")

        # model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment__film_updated__2_layers")
        # model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment__film_updated__1_layer_2_per_film_plus_0")
        # evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[256], n_seeds=5)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512, 1024], n_seeds=1)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512], n_seeds=10)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512, 1024], n_seeds=9)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512], n_seeds=9)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512, 1024], n_seeds=10)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512], n_seeds=10)

        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512, 1024], n_seeds=20)
        evaluate_latent_dim_experiment(model_dir=model_dir, latent_dims=[32, 64, 128, 256, 512], n_seeds=20)

    if conditioning_experiment:
        for latent_dim in [256]:
            print("Run conditioning experiment ...")
            model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning_2")
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            evaluate_conditioning_experiment(model_dir, latent_dim=latent_dim, dec_hidden_dim=256, n_seeds=10)
            
        
    
    if elbo_vs_geco_experiment:
        print("Run ELBO vs. GECO experiment experiment ...")
        model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning_2")
        evaluate_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=1)
        evaluate_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=5)
        evaluate_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        evaluate_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        evaluate_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        
            
            




    args = {}
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    args["n_epochs"] = 100
    args["seed"] = 0


    # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_32")
    # model_path_3 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[3]")
    # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[31]")

    for batch_size in [64]:
        for n_train in [10_000]:  # [5_000, 10_000, 20_000, 30_000, 40_000, 60_000, 80_000, 100_000, 120_000]:
            for lr in [1e-4]:  # [1e-3, 1e-4]:
                for model_epoch in [850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, ]:  # [850, 900, 950]:  
                # for gen_path in [file for file in model_path_3.glob('*modelWeights*') if file.is_file()]:
                #     print(f"\ngen_path = {gen_path}")

                #     file_name = gen_path.name
                #     print(f"file_name = {file_name}")

                #     gen_path_3 = str(gen_path)
                #     gen_path_31 = str(model_path_31 / file_name)
                #     print(f"gen_path_3 = {gen_path_3}")
                #     print(f"gen_path_31 = {gen_path_31}")
                    
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")

                    args["output_dir"] = (
                        f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_{batch_size}__lr_{lr}__train_on_only_synthetic_{timestamp}"
                    )
                    # args["generative_model_args_path"] = (
                    #     "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_20240801_082756/args.json"
                    # )
                    args["generative_model_weights_path"] = [
                        # gen_path_3,
                        # gen_path_31
                        # VAES
                        # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103730/modelWeights",  # cls 3
                        # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103916/modelWeights",  # cls 31

                        # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151151/modelWeights",  # cls 3
                        # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151204/modelWeights",  # cls 31
                        
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120",  # cls 3
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_180210/modelWeights_epoch_120",  # cls 31

                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110",  # cls [3, 31]


                        # # VAE experiment latent_dim
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_512_cond_film/modelWeights_epoch_{model_epoch}"

                        
                        # # VAE experiment conditioning
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_concat__phoneme_cls_[3, 31]_dec_emb_dim_128__dec_hidden_dim_512/modelWeights_epoch_{model_epoch}"  # cond experiment
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_film__phoneme_cls_[3, 31]/modelWeights_epoch_{model_epoch}"


                        # # VAE experiment elbo vs geco
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_elbo_vs_geco/VAE__loss_elbo__beta_init_0.001/modelWeights_epoch_{model_epoch}"  # cond experiment


                        # GAN
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173938__phoneme_cls_3_31/modelWeights_1000"  # GAN classes 3, 31
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173804__phoneme_cls_3_31/modelWeights_{model_epoch}"
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{model_epoch}"
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{model_epoch}"

                        f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240829_182616__phoneme_cls_3_31/modelWeights_{model_epoch}"  # GAN concat 32
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_120647__phoneme_cls_3/modelWeights_{model_epoch}",
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_235138__phoneme_cls_31/modelWeights_{model_epoch}"

                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_234931__phoneme_cls_3/modelWeights_{model_epoch}",
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_101505__phoneme_cls_31/modelWeights_{model_epoch}"

                    ]
                    args["generative_model_n_samples_train"] = n_train
                    args["generative_model_n_samples_val"] = 2_000
                    args["generative_model_n_samples_test"] = 2_000
                    print(args["generative_model_weights_path"])

                    args["input_shape"] = (128, 8, 8)
                    args["lr"] = lr
                    args["batch_size"] = batch_size
                    args["transform"] = "softsign"
                    args["patience"] = 20
                    args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                    args["correctness_value"] = ["C"]

                    print(
                        "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
                    )

                    main(args)
