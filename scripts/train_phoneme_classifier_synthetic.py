import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
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
from neural_decoder.model_phoneme_classifier import train_phoneme_classifier, PhonemeClassifier
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
    # print("Training phoneme classifier with the following arguments:")
    # for k, v in args.items():
    #     print(f"{k}: {v}")

    set_seeds(args["seed"])

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    
    # load generative models
    if "generative_model_weights_path" in args.keys():
        if isinstance(args["generative_model_weights_path"], list):
            gen_models = []
            for weights_path in args["generative_model_weights_path"]:
                model = load_t2b_gen_model(weights_path=weights_path)
                gen_models.append(model)
        else:
            gen_models = [load_t2b_gen_model(weights_path=args["generative_model_weights_path"])]

        output = train_phoneme_classifier(
            gen_models=gen_models,
            n_samples_train_syn=args["generative_model_n_samples_train"],
            n_samples_val=args["generative_model_n_samples_val"],
            n_samples_test=args["generative_model_n_samples_test"],
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

    elif "train_set_path" in args.keys():
        class_counts = [
            4841, 7058, 27635, 3298, 2566, 7524, 4674, 2062, 11389, 9501,
            6125, 6573, 4027, 4259, 3315, 5505, 15591, 10434, 2194, 9755,
            13949, 9138, 18297, 3411, 3658, 1661, 6034, 11435, 11605, 2815,
            23188, 2083, 1688, 8414, 6566, 6633, 3707, 7403, 7807
        ] # fmt: on
        if len(phoneme_classes) < len(class_counts):
            class_counts = [class_counts[i-1] for i in phoneme_classes]

        if args["class_weights"] == "sqrt":
            class_weights = torch.tensor(1.0 / np.sqrt(np.array(class_counts))).float()
        elif args["class_weights"] == "inv":
            class_weights = torch.tensor(1.0 / np.array(class_counts)).float()
        else:
            class_weights = None

        # load train dataloader
        train_dl = get_data_loader(
            data=load_pkl(args["train_set_path"]),
            batch_size=args["batch_size"],
            shuffle=False,
            collate_fn=None,
            dataset_cls=PhonemeDataset,
            phoneme_ds_filter=phoneme_ds_filter,
            class_weights=class_weights,
            transform=transform,
        )
        train_dl.name = "train-real"

        n_classes = len(phoneme_classes)
        args["n_classes"] = n_classes
        model = PhonemeClassifier(classes=phoneme_classes, input_shape=args["input_shape"]).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
        
        if n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        with open(out_dir / "args", "wb") as file:
            pickle.dump(args, file)
        with open(out_dir / "args.json", "w") as file:
            json.dump(args, file, indent=4)

        output = model.train_model(
            train_dl=train_dl,
            val_dls=[val_dl_real],
            test_dl=[test_dl_real],
            n_classes=n_classes,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            out_dir=out_dir,
            n_epochs=args["n_epochs"],
            patience=args["patience"],
            model_classes=phoneme_classes,
        )
        return output 


def evaluate_vae_latent_dim_experiment(
    model_dir: Path, 
    file_name_pattern: str,
    pattern_latent_dims: list, 
    pattern_seeds: list,
    plot_dir: Path,
    syn_n_samples: int, 
    n_seeds: int = 5, 
    bootstrap_iters: int = 1_000,
    metric: str = "auroc",
) -> None:
    assert "{latent_dim}" in file_name_pattern, "'file_name_pattern' must contain '{latent_dim}' placeholder"

    plot_dir.mkdir(parents=True, exist_ok=True)

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
    args["patience"] = 15
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    _aurocs = []
    _sems = []
    x_labels = []

    for latent_dim in pattern_latent_dims:
        seed_aurocs = []

        for pattern_seed in pattern_seeds:
            sub_dir = model_dir / file_name_pattern.format(latent_dim=latent_dim, seed=pattern_seed)  # f"VAE__latent_dim_{latent_dim}_cond_film"
            matching_files = list(sub_dir.glob("modelWeights*"))
            assert len(matching_files) == 1, "There are multiple modelWeights files in the given directory."

            weights_file = matching_files[0]
            print(f"weights_file = {weights_file}")

            # creat out dir
            out_dir = sub_dir / "phoneme_classifier"
            out_dir.mkdir(exist_ok=True)

            for seed in range(n_seeds):
                args["seed"] = seed
                args["output_dir"] = (
                    f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
                )
                args["generative_model_weights_path"] = str(weights_file)
                args["generative_model_n_samples_train"] = syn_n_samples
                args["generative_model_n_samples_val"] = 5_000
                args["generative_model_n_samples_test"] = 5_000  # add testing on synthetic data

                output = main(args)
                print(f"output['test_acc'] = {output['test_acc']}")
                print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
                print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
                
                auroc, _ = compute_auroc_with_stderr(
                    y_true=output["test_y_true"], 
                    y_pred=output["test_y_pred"], 
                    n_iters=bootstrap_iters, 
                )
                print(f"output['test_y_true'].shape = {output['test_y_true'].shape}")
                print(f"output['test_y_pred'].shape = {output['test_y_pred'].shape}")
                seed_aurocs.append(auroc)

                # make subdir for this random seed and n samples
                sub_out_dir = out_dir / f"phoneme_classifier__syn_n_samples_{syn_n_samples}__seed_{seed}"
                sub_out_dir.mkdir(exist_ok=True)
                data = {
                    "y_true": output["test_y_true"].tolist(),
                    "y_pred": output["test_y_pred"].tolist(),
                    "y_true_syn": output["test_y_true_syn"].tolist(),
                    "y_pred_syn": output["test_y_pred_syn"].tolist(),
                    "seed": seed,
                    "generative_model_n_samples_train": syn_n_samples
                }
                with open(sub_out_dir / "ouput.json", "w") as json_file:
                    json.dump(data, json_file)
                    print(f"Saved output to {sub_out_dir / 'ouput.json'}")

        
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=n_seeds, replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        _auroc = np.mean(seed_aurocs)
        _sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)

        _aurocs.append(_auroc)
        _sems.append(_sem)

        x_labels.append(latent_dim)

        colors = sns.color_palette("Set2", 3)[2]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        plot_aurocs_with_error_bars(
            aurocs=_aurocs, 
            errs=_sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions",
            colors=colors,
        )
        plot_aurocs_with_error_bars(
            aurocs=_aurocs, 
            errs=_sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_with_annotations__syn_n_samples_{syn_n_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions",
            colors=colors,
            add_annotations=True
        )


def evaluate_vae_conditioning_experiment(
    model_dir: Path, 
    latent_dim: int, 
    dec_hidden_dim: int, 
    pattern_seeds: list,
    plot_dir: Path,
    syn_n_samples: int,
    n_seeds: int = 10,
    bootstrap_iters: int = 1_000,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    
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
    args["patience"] = 15
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    _aurocs = []
    all_seed_aurocs = []
    _sems = []
    x_labels = []

    name2dir = {
        "Separate Models": [
            f"vae__conditioning_None__phoneme_cls_3__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
            f"vae__conditioning_None__phoneme_cls_31__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}"
        ],

        "FiLM": f"vae__conditioning_film__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",

        "Concat 2": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_2__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 4": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_4__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 8": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_8__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 16": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_16__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 32": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_32__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 64": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_64__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 128": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_128__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
    }
    for model_name, weight_dir in name2dir.items():
        seed_aurocs = []

        for pattern_seed in pattern_seeds:
            if isinstance(weight_dir, str):
                sub_dir = model_dir / weight_dir.format(pattern_seed=pattern_seed)
                matching_files = list(sub_dir.glob("modelWeights*"))
                assert len(matching_files) == 1, f"There are multiple modelWeights files in the given directory, expected one: {matching_files}"
                weights_file = str(matching_files[0])

                # creat out dir
                out_dir = sub_dir / "phoneme_classifier"
                out_dir.mkdir(exist_ok=True)

            elif isinstance(weight_dir, list):
                weights_file = []
                for sub_dir in weight_dir:
                    matching_files = list((model_dir / sub_dir.format(pattern_seed=pattern_seed)).glob("modelWeights*"))
                    assert len(matching_files) == 1, f"There are {len(matching_files)} modelWeights files in the given directory, expected one: {matching_files}"
                    weights_file.append(str(matching_files[0]))

                # creat out dir
                out_dir = model_dir / weight_dir[0].format(pattern_seed=pattern_seed) / "phoneme_classifier"
                out_dir.mkdir(exist_ok=True)

            for seed in range(n_seeds):
                args["seed"] = seed
                args["output_dir"] = (
                    f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
                )
                
                args["generative_model_weights_path"] = weights_file
                args["generative_model_n_samples_train"] = syn_n_samples
                args["generative_model_n_samples_val"] = 5_000
                args["generative_model_n_samples_test"] = 5_000

                output = main(args)
                print(f"output['test_acc'] = {output['test_acc']}")
                print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
                print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
                
                auroc, _ = compute_auroc_with_stderr(
                    y_true=output["test_y_true"], 
                    y_pred=output["test_y_pred"], 
                    n_iters=bootstrap_iters, 
                )
                seed_aurocs.append(auroc)

                # make subdir for this random seed and n samples
                sub_out_dir = out_dir / f"phoneme_classifier__syn_n_samples_{syn_n_samples}__seed_{seed}"
                sub_out_dir.mkdir(exist_ok=True)
                data = {
                    "y_true": output["test_y_true"].tolist(),
                    "y_pred": output["test_y_pred"].tolist(),
                    "y_true_syn": output["test_y_true_syn"].tolist(),
                    "y_pred_syn": output["test_y_pred_syn"].tolist(),
                    "seed": seed,
                    "generative_model_n_samples_train": syn_n_samples
                }
                with open(sub_out_dir / "ouput.json", "w") as json_file:
                    json.dump(data, json_file)
                    print(f"Saved output to {sub_out_dir / 'ouput.json'}")


        all_seed_aurocs.append(seed_aurocs)
        
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=n_seeds, replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        _auroc = np.mean(seed_aurocs)
        _sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)

        _aurocs.append(_auroc)
        _sems.append(_sem)

        x_labels.append(model_name)


def plot_venn3(label2values: Dict[str, np.ndarray], out_file: Path, title: str = "Venn Diagram") -> None:
    assert len(label2values) == 3

    indices = [set(v) for v in label2values.values()]
    labels = list(label2values.keys())
    indices = [indices[i] for i in [1, 0, 2]]
    labels = [labels[i] for i in [1, 0, 2]]

    venn3(indices, set_labels=labels)

    plt.title(title)
    plt.savefig(out_file)
    plt.close()


def get_miclassified_indices(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_pred_binary = y_pred_binary.flatten()
    assert y_pred_binary.shape == y_true.shape
    
    misclassified_indices = np.where(y_true != y_pred_binary)[0]    
    return misclassified_indices


def evaluate_final_experiment(
    vae_dir: Path, 
    vae_pattern: str,
    gan_dir: Path, 
    gan_pattern: str, 
    pattern_seeds: list, 
    syn_n_samples: int, 
    n_seeds: int = 10, 
    bootstrap_iters: int = 10,
    prefix=""
):
    
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
    args["patience"] = 15
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    aurocs = []
    sems = []
    syn_aurocs = []
    syn_sems = []
    all_seed_aurocs = []
    all_test_trues = []
   
    x_labels = []
    
    model_names = ["Real data", "GAN data", "VAE data"]
    thresholds = [0.4, 0.5, 0.6,]
    label2miscls = {t: {m: [] for m in model_names} for t in thresholds}

    for model_dir, model_name in zip([None, gan_dir, vae_dir], model_names):
        print(f"\nmodel_name = {model_name}")
        seed_aurocs = []
        syn_seed_aurocs = []
        model_seeds = [None] if model_dir is None else pattern_seeds

        for i, p_seed in enumerate(model_seeds):
            if model_dir is not None:
                if "GAN" in model_name:
                    sub_dir = model_dir / gan_pattern.format(seed=p_seed)
                elif "VAE" in model_name:
                    sub_dir = model_dir / vae_pattern.format(seed=p_seed)

                weights_file = list(sub_dir.glob("modelWeights_*_best_auroc"))
                if len(weights_file) == 0:
                    weights_file = list(sub_dir.glob("modelWeights_*"))
                assert len(weights_file) == 1, f"weights_file = {weights_file}"
                weights_file = weights_file[0]
                

                if isinstance(weights_file, str):
                    weights_file = str(weights_file)
                elif isinstance(weights_file, list):
                    weights_file = [str(f) for f in weights_file]
                
                args["generative_model_weights_path"] = weights_file
                args["generative_model_n_samples_train"] = syn_n_samples
                args["generative_model_n_samples_val"] = 5_000
                args["generative_model_n_samples_test"] = 5_000

                # creat out dir
                out_dir = sub_dir / "phoneme_classifier"
                out_dir.mkdir(exist_ok=True)

            elif model_dir is None:
                args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
                args["class_weights"] = "sqrt"
                args.pop("generative_model_weights_path", None)
                args.pop("generative_model_n_samples_train", None)
                args.pop("generative_model_n_samples_val", None)
                args.pop("generative_model_n_samples_test", None)

                # creat out dir
                out_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments") / "classifier_trained_on_real" / "phoneme_classifier"
                out_dir.mkdir(exist_ok=True)

            for j, seed in enumerate(range(n_seeds)):
                args["seed"] = seed
                args["output_dir"] = (
                    f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
                )

                output = main(args)
                print(f"output['test_acc'] = {output['test_acc']}")
                print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
                print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
                
                auroc, _ = compute_auroc_with_stderr(
                    y_true=output["test_y_true"], 
                    y_pred=output["test_y_pred"], 
                    n_iters=bootstrap_iters, 
                )
                seed_aurocs.append(auroc)

                if model_dir is not None:
                    print(f"output.keys() = {output.keys()}")
                    syn_auroc, _ = compute_auroc_with_stderr(
                        y_true=output["test_y_true_syn"], 
                        y_pred=output["test_y_pred_syn"], 
                        n_iters=bootstrap_iters, 
                    )
                    syn_seed_aurocs.append(syn_auroc)

                if i == 0 and j == 0:
                    for t in thresholds:
                        miscls_indices = get_miclassified_indices(
                            y_true=output["test_y_true"],  
                            y_pred=output["test_y_pred"],
                            threshold=t,
                        )
                        label2miscls[t][model_name] = miscls_indices
                        all_test_trues.append(output["test_y_true"])

                # make subdir for this random seed and n samples
                sub_out_dir = out_dir / f"phoneme_classifier__syn_n_samples_{syn_n_samples}__seed_{seed}"
                sub_out_dir.mkdir(exist_ok=True)
                if model_dir is None:
                    data = {
                        "y_true": output["test_y_true"].tolist(),
                        "y_pred": output["test_y_pred"].tolist(),
                        "seed": seed,
                        "generative_model_n_samples_train": syn_n_samples
                    }
                else:
                    data = {
                        "y_true": output["test_y_true"].tolist(),
                        "y_pred": output["test_y_pred"].tolist(),
                        "y_true_syn": output["test_y_true_syn"].tolist(),
                        "y_pred_syn": output["test_y_pred_syn"].tolist(),
                        "seed": seed,
                        "generative_model_n_samples_train": syn_n_samples
                    }
                with open(sub_out_dir / "ouput.json", "w") as json_file:
                    json.dump(data, json_file)
                    print(f"Saved output to {sub_out_dir / 'ouput.json'}")

        all_seed_aurocs.append(seed_aurocs)
            
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=len(seed_aurocs), replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        auroc = np.mean(seed_aurocs)
        sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        aurocs.append(auroc)
        sems.append(sem)

        if model_dir is not None:
            syn_bootstrapped_aurocs = []
            for _ in range(bootstrap_iters):
                sample = np.random.choice(syn_seed_aurocs, size=len(seed_aurocs), replace=True)
                syn_bootstrapped_aurocs.append(np.mean(sample))

            syn_auroc = np.mean(syn_seed_aurocs)
            syn_sem = np.std(syn_bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
            syn_aurocs.append(syn_auroc)
            syn_sems.append(syn_sem)

        x_labels.append(model_name)

    colors = sns.color_palette("Set2", len(x_labels))

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    plot_dir = ROOT_DIR / "evaluation" / "experiments" / "final_experiment" / "plots"

    for i, figsize in enumerate([(8, 6), (9, 6), (10, 6)]):

        plot_aurocs_with_error_bars(
            aurocs=aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_{i}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}.png",
            xlabel="Training Data",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data \n Evaluated on real data",
            colors=colors,
            figsize=figsize,
        )
        
        plot_aurocs_with_error_bars(
            aurocs=aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_{i}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_with_annotations.png",
            xlabel="Training Data",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data \n Evaluated on real data",
            colors=colors,
            figsize=figsize,
            add_annotations=True
        )
        plot_aurocs_with_error_bars(
            aurocs=[syn_aurocs[0]], 
            errs=[syn_sems[0]], 
            x_labels=[x_labels[1]], 
            out_file=plot_dir / f"{timestamp}__GAN_SEM_aurocs_with_err__seed_{n_seeds}_SYN__syn_n_samples_{syn_n_samples}.png",
            xlabel="Conditiong mechanism",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data \n Evaluated on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs),
            figsize=(11, 6),
        )
        plot_aurocs_with_error_bars(
            aurocs=[syn_aurocs[1]], 
            errs=[syn_sems[1]], 
            x_labels=[x_labels[2]], 
            out_file=plot_dir / f"{timestamp}__VAE_SEM_aurocs_with_err__seed_{n_seeds}_SYN_with_annotations__syn_n_samples_{syn_n_samples}.png",
            xlabel="Conditiong mechanism",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data \n Evaluated on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs),
            add_annotations=True,
            figsize=(11, 6),
        )
    
    model_auroc_dict = dict(zip(x_labels, all_seed_aurocs))
    output_file = plot_dir / f"{timestamp}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}.json"
    with open(output_file, "w") as f:
        json.dump(model_auroc_dict, f, indent=4)



    # aurocs.insert(3, syn_aurocs[0])
    # sems.insert(3, syn_sems[0])
    # x_labels.insert(3, "GAN data ")
    # colors.insert(3, colors[1])

    # aurocs.insert(4, syn_aurocs[1])
    # sems.insert(4, syn_sems[1])
    # x_labels.insert(4, "VAE data ")
    # colors.insert(4, colors[2])

    # hatches=['', '', '', '|', '|']

    # print(f"\naurocs = {aurocs}")
    # print(f"sems = {sems}")
    # print(f"x_labels = {x_labels}")
    # print(f"hatches = {hatches}")
    # print(f"colors = {colors}")

    syn_x_lables = ["GAN data", "VAE data"]


    plot_aurocs_with_error_bars(
        aurocs=[aurocs, syn_aurocs], 
        errs=[sems, syn_sems], 
        x_labels=[x_labels, syn_x_lables], 
        out_file=plot_dir / f"{timestamp}_ALL_5__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors[1:]],
        hatches=[['', '', ''], ['|', '|']],
        figsize=(11, 6),
        y_max_val=1.0002
    )
    
    plot_aurocs_with_error_bars(
        aurocs=[aurocs, syn_aurocs], 
        errs=[sems, syn_sems], 
        x_labels=[x_labels, syn_x_lables], 
        out_file=plot_dir / f"{timestamp}_ALL_5__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL_with_annotations.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors[1:]],
        hatches=[['', '', ''], ['|', '|']],
        figsize=(11, 6),
        add_annotations=True,
        y_max_val=1.0002
    )



    for t in thresholds:
        plot_venn3(label2miscls[t], plot_dir / "venn_plots" / f"venn3_{timestamp}__{prefix}__threshold_{t}.png", title=f"Overlap of Misclassified Samples Across Models \n(threshold = {t})")

    print(f"np.array_equal(all_test_trues[0], all_test_trues[1]) = {np.array_equal(all_test_trues[0], all_test_trues[1])}")
    print(f"np.array_equal(all_test_trues[0], all_test_trues[2]) = {np.array_equal(all_test_trues[0], all_test_trues[2])}")


def evaluate_gan_conditioning_experiment(
    concat_weights_file: Path, 
    film_weights_file: Path, 
    separate_weights_file: List[Path], 
    n_seeds: int = 1, 
    bootstrap_iters: int = 1_000
):
    
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
    args["patience"] = 15
    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]

    _aurocs = []
    _sems = []
    syn_aurocs = []
    syn_sems = []
    x_labels = []

    weights_files = [separate_weights_file, film_weights_file, concat_weights_file]
    model_names = ["Separate models", "FiLM", "Concat"]
    
    for weights_file, model_name in zip(weights_files, model_names):
        print(f"model_name = {model_name}")

        if isinstance(weights_file, str):
            weights_file = str(weights_file)
        elif isinstance(weights_file, list):
            weights_file = [str(f) for f in weights_file]
        
        args["generative_model_weights_path"] = weights_file
        syn_n_samples = 10_000
        args["generative_model_n_samples_train"] = syn_n_samples
        args["generative_model_n_samples_val"] = 5_000
        args["generative_model_n_samples_test"] = 5_000

        if weights_file is None:
            args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
            args["class_weights"] = "sqrt"
            del args["generative_model_weights_path"]
            del args["generative_model_n_samples_train"]
            del args["generative_model_n_samples_val"]
            del args["generative_model_n_samples_test"]

        seed_aurocs = []
        syn_seed_aurocs = []

        for seed in range(n_seeds):
            args["seed"] = seed
            args["output_dir"] = (
                f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
            )

            output = main(args)
            print(f"output['test_acc'] = {output['test_acc']}")
            print(f"output['test_auroc_macro'] = {output['test_auroc_macro']}")
            print(f"output['test_auroc_micro'] = {output['test_auroc_micro']}")
            
            auroc, _ = compute_auroc_with_stderr(
                y_true=output["test_y_true"], 
                y_pred=output["test_y_pred"], 
                n_iters=bootstrap_iters, 
            )
            seed_aurocs.append(auroc)

            syn_auroc, _ = compute_auroc_with_stderr(
                y_true=output["test_y_true_syn"], 
                y_pred=output["test_y_pred_syn"], 
                n_iters=bootstrap_iters, 
            )
            syn_seed_aurocs.append(syn_auroc)
            
        
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=n_seeds, replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        _auroc = np.mean(seed_aurocs)
        _sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        _aurocs.append(_auroc)
        _sems.append(_sem)

        syn_bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(syn_seed_aurocs, size=len(seed_aurocs), replace=True)
            syn_bootstrapped_aurocs.append(np.mean(sample))

        syn_auroc = np.mean(syn_seed_aurocs)
        syn_sem = np.std(syn_bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        syn_aurocs.append(syn_auroc)
        syn_sems.append(syn_sem)

        x_labels.append(model_name)


    colors = sns.color_palette("Set2", len(x_labels))[1]

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # plot_aurocs_with_error_bars(
    #     aurocs=_aurocs, 
    #     errs=_sems, 
    #     x_labels=x_labels, 
    #     out_file=ROOT_DIR / "evaluation" / "gan" / "final_models" / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}.png",
    #     xlabel="Conditiong mechanism",
    #     title="AUROC performance of GANs \nwith varying conditioning mechanisms",
    #     colors=colors,
    #     figsize=(8, 6),
    # )
    plot_dir = ROOT_DIR / "evaluation" / "experiments" / "gan_conditioning"


    plot_aurocs_with_error_bars(
        aurocs=[_aurocs, syn_aurocs], 
        errs=[_sems, syn_sems], 
        x_labels=[x_labels, x_labels], 
        out_file=plot_dir / f"{timestamp}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL_1.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors],
        hatches=[['', '', ''], ['|', '|', '|']],
        figsize=(11, 6),
        y_max_val=1.0002,
        y_min_val=0.8
    )
    
    plot_aurocs_with_error_bars(
        aurocs=[_aurocs, syn_aurocs], 
        errs=[_sems, syn_sems], 
        x_labels=[x_labels, x_labels], 
        out_file=plot_dir / f"{timestamp}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL_1_with_annotations.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors],
        hatches=[['', '', ''], ['|', '|', '|']],
        figsize=(11, 6),
        add_annotations=True,
        y_max_val=1.0002,
        y_min_val=0.8
    )
    
    print(f"\nlen(_aurocs) = {len(_aurocs)}")
    print(f"_aurocs = {_aurocs}")
    _aurocs = [0.869, 0.872, 0.861]
    print(f"\nlen(_aurocs) = {len(_aurocs)}")
    print(f"_aurocs = {_aurocs}")


    plot_aurocs_with_error_bars(
        aurocs=[_aurocs, syn_aurocs], 
        errs=[_sems, syn_sems], 
        x_labels=[x_labels, x_labels], 
        out_file=plot_dir / f"{timestamp}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL_2.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors],
        hatches=[['', '', ''], ['|', '|', '|']],
        figsize=(11, 6),
        y_max_val=1.0002,
        y_min_val=0.8
    )
    
    plot_aurocs_with_error_bars(
        aurocs=[_aurocs, syn_aurocs], 
        errs=[_sems, syn_sems], 
        x_labels=[x_labels, x_labels], 
        out_file=plot_dir / f"{timestamp}__SEM_aurocs_with_err__seed_{n_seeds}__syn_n_samples_{syn_n_samples}_ALL_2_with_annotations.png",
        xlabel="Training Data",
        title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
        sub_titles=["Evaluated on real data", "Evaluated on synthetic data"],
        colors=[colors, colors],
        hatches=[['', '', ''], ['|', '|', '|']],
        figsize=(11, 6),
        add_annotations=True,
        y_max_val=1.0002,
        y_min_val=0.8
    )



if __name__ == "__main__":
    print("in main ...")

    vae_latent_dim_experiment = False
    vae_conditioning_experiment = False
    vae_elbo_vs_geco_experiment = False
    
    gan_conditioning_experiment = True
    
    final_experiment = False

    if vae_latent_dim_experiment:
        for dhd in [256, 512,]:
            for n_seeds in [10]:
                for syn_n_samples in [10_000, 30_000, 50_000, ]:
                    evaluate_vae_latent_dim_experiment(
                        model_dir=Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_latent_dim_cond_bn_True__dhd_{dhd}"), 
                        file_name_pattern="vae__latent_dim_{latent_dim}__seed_{seed}", 
                        pattern_latent_dims=[32, 64, 128, 256, 512, 1024], 
                        pattern_seeds=list(range(10)),
                        n_seeds=n_seeds,
                        syn_n_samples=syn_n_samples,
                        plot_dir= ROOT_DIR / "evaluation" / "experiments" / "vae_latent_dim_cond_bn_True" / "plots"
                    )
    
    if vae_conditioning_experiment:
        for n_seeds in [10]:
            for latent_dim in [256,]:
                for dec_hidden_dim in [512,]: # TODO 512
                    for syn_n_samples in [30_000, 50_000, ]:
                        print("Run conditioning experiment ...")
                        model_dir = Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_conditioning_cond_bn_True/ld_{latent_dim}_dhd_{dec_hidden_dim}")
                        evaluate_vae_conditioning_experiment(
                            model_dir=model_dir, 
                            latent_dim=latent_dim,
                            dec_hidden_dim=dec_hidden_dim,
                            pattern_seeds=list(range(10)),
                            n_seeds=n_seeds,
                            syn_n_samples=syn_n_samples,
                            plot_dir=ROOT_DIR / "evaluation" / "experiments" / "vae_conditioning_cond_bn_True" / f"ld_{latent_dim}_dhd_{dec_hidden_dim}" / "plots"
                        )
                    
    if vae_elbo_vs_geco_experiment:
        print("Run ELBO vs. GECO experiment experiment ...")
        model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning_2")
        evaluate_vae_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=1)
        evaluate_vae_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=5)
        evaluate_vae_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        evaluate_vae_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        evaluate_vae_conditioning_experiment(model_dir, latent_dim=256, dec_hidden_dim=256, n_seeds=10)
        

    if gan_conditioning_experiment:
        model_root_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs")
        for n_seeds in [5]:
            evaluate_gan_conditioning_experiment(
                separate_weights_file=[
                    "/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_012728__phoneme_cls_3/modelWeights_1480",
                    "/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232908__phoneme_cls_31/modelWeights_1480"
                ],
                film_weights_file="/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_100458__phoneme_cls_3_31/modelWeights_920",
                concat_weights_file="/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240901_092827__phoneme_cls_3_31/modelWeights_1440",
                n_seeds=n_seeds,
            )

    if final_experiment: 
        for n_seeds in [3,]:
            for syn_n_samples in [10_000]:  #, 50_000, 40_000, 20_000, 5_000, 1_000]:
                for ld in [256]:
                    for dhd in [512]:
                        for gan_ld in [512]:

                            evaluate_final_experiment(
                                # vae_dir=Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_latent_dim_cond_bn_True__dhd_512"),
                                vae_dir=Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_conditioning_cond_bn_True/ld_{ld}_dhd_{dhd}"),
                                gan_dir=Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/gan_conditioning/ld_{gan_ld}"),
                                vae_pattern=f"vae__conditioning_film__phoneme_cls_3_31__latent_dim_{ld}__dec_emb_dim_None__dec_hidden_dim_{dhd}__seed_{{seed}}",
                                gan_pattern=f"gan__conditioning_film__dec_emb_dim_None__phoneme_cls_3_31__seed_{{seed}}",
                                pattern_seeds=range(10),
                                n_seeds=n_seeds,
                                syn_n_samples=syn_n_samples,
                                prefix=f"vae_ld_{ld}_dhd_{dhd}"
                            
                            )

    

                            # # VAE (256, 512), GAN (512
                            # evaluate_final_experiment(
                            #     # vae_weights_file="/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment_2/VAE__latent_dim_256_cond_film/modelWeights_epoch_83", 
                            #     vae_dir=Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_latent_dim_cond_bn_True__dhd_512"),
                            #     gan_dir=Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/gan_conditioning/ld_512"),
                            #     vae_pattern="vae__latent_dim_256__seed_{seed}",
                            #     gan_pattern="gan__conditioning_film__dec_emb_dim_None__phoneme_cls_3_31__seed_{seed}",
                            #     pattern_seeds=range(1),
                            #     n_seeds=n_seeds,
                            #     syn_n_samples=syn_n_samples
                            
                            #     # gan_weights_file="/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_100458__phoneme_cls_3_31/modelWeights_920",
                            #     # [
                            #     #     "/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_012728__phoneme_cls_3/modelWeights_1480",
                            #     #     "/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232908__phoneme_cls_31/modelWeights_1480"
                            #     # ],
                            
                            # )


    pass

    # args = {}
    # args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    # args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    # args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    # args["n_epochs"] = 100
    


    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_32")
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[3]")
    # # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[31]")
    
    # # # lr_g=0.0001 lr_d=5e-5
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_233528__phoneme_cls_3")
    # # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232314__phoneme_cls_31")

    # # # lr_g=1e-5 lr_d=5e-5
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_012728__phoneme_cls_3")
    # # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232908__phoneme_cls_31")

    # # # lr_g=5e-6 lr_d=1e-5  A
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240901_095835__phoneme_cls_3")
    # # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232348__phoneme_cls_31")

    # # # lr_g=5e-6 lr_d=1e-5  B
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240901_133439__phoneme_cls_3")
    # # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_232348__phoneme_cls_31")


    # # best GAN concat
    # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240901_092827__phoneme_cls_3_31")
    
    # # best GAN film
    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240902_100458__phoneme_cls_3_31")

    # print(f"model_path = {model_path}")

    # for seed in [1,]:
    #     for batch_size in [64]:
    #         for n_train in [10_000]:  # [5_000, 10_000, 20_000, 30_000, 40_000, 60_000, 80_000, 100_000, 120_000]:
    #             for lr in [1e-4]:  # [1e-3, 1e-4]:
    #                 # for model_epoch in list(range(850, 2301, 50)):  # [850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, ]:  # [850, 900, 950]:  
    #                 for gen_path in [file for file in model_path.glob('*modelWeights_*') if file.is_file()]:
    #                     # print(f"\ngen_path = {gen_path}")

    #                     # file_name = gen_path.name
    #                     # print(f"file_name = {file_name}")

    #                     # gen_path_3 = str(gen_path)
    #                     # gen_path_31 = str(model_path_31 / file_name)
    #                     # print(f"gen_path_3 = {gen_path_3}")
    #                     # print(f"gen_path_31 = {gen_path_31}")
                        
    #                     args["seed"] = seed
    #                     now = datetime.now()
    #                     timestamp = now.strftime("%Y%m%d_%H%M%S")

    #                     args["output_dir"] = (
    #                         f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_{batch_size}__lr_{lr}__train_on_only_synthetic_{timestamp}"
    #                     )
    #                     # args["generative_model_args_path"] = (
    #                     #     "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_20240801_082756/args.json"
    #                     # )
    #                     args["generative_model_weights_path"] = [
    #                         gen_path
    #                         # gen_path_3,
    #                         # gen_path_31
    #                         # VAES
    #                         # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103730/modelWeights",  # cls 3
    #                         # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103916/modelWeights",  # cls 31

    #                         # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151151/modelWeights",  # cls 3
    #                         # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151204/modelWeights",  # cls 31
                            
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120",  # cls 3
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_180210/modelWeights_epoch_120",  # cls 31

    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110",  # cls [3, 31]


    #                         # # VAE experiment latent_dim
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_512_cond_film/modelWeights_epoch_{model_epoch}"

                            
    #                         # # VAE experiment conditioning
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_concat__phoneme_cls_[3, 31]_dec_emb_dim_128__dec_hidden_dim_512/modelWeights_epoch_{model_epoch}"  # cond experiment
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_film__phoneme_cls_[3, 31]/modelWeights_epoch_{model_epoch}"


    #                         # # VAE experiment elbo vs geco
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_elbo_vs_geco/VAE__loss_elbo__beta_init_0.001/modelWeights_epoch_{model_epoch}"  # cond experiment


    #                         # GAN
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173938__phoneme_cls_3_31/modelWeights_1000"  # GAN classes 3, 31
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173804__phoneme_cls_3_31/modelWeights_{model_epoch}"
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{model_epoch}"
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{model_epoch}"

    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240829_182616__phoneme_cls_3_31/modelWeights_{model_epoch}"  # GAN concat 32

    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240901_094401__phoneme_cls_3_31/modelWeights_{model_epoch}"

    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_120647__phoneme_cls_3/modelWeights_{model_epoch}",
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_235138__phoneme_cls_31/modelWeights_{model_epoch}"

    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_234931__phoneme_cls_3/modelWeights_{model_epoch}",
    #                         # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_101505__phoneme_cls_31/modelWeights_{model_epoch}"

    #                     ]
    #                     args["generative_model_n_samples_train"] = n_train
    #                     args["generative_model_n_samples_val"] = 5_000
    #                     args["generative_model_n_samples_test"] = 5_000
    #                     print(args["generative_model_weights_path"])

    #                     args["input_shape"] = (128, 8, 8)
    #                     args["lr"] = lr
    #                     args["batch_size"] = batch_size
    #                     args["transform"] = "softsign"
    #                     args["patience"] = 15
    #                     args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
    #                     args["correctness_value"] = ["C"]

    #                     print(
    #                         "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
    #                     )

    #                     main(args)
