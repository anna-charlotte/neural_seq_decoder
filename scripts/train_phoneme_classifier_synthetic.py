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
from neural_decoder.model_phoneme_classifier import (
    PhonemeClassifier,
    train_phoneme_classifier,
)
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
from utils import load_pkl, set_seeds


def plot_vae_decoder_weights(vae) -> None:
    fc_weights = vae.decoder.fc[0].weight.data.detach().cpu().numpy()

    conv_layers = [layer for layer in vae.decoder.model if isinstance(layer, nn.ConvTranspose2d)]
    conv_weights = [layer.weight.data.cpu().numpy() for layer in conv_layers]

    # Plotting the heatmaps of the weights
    plt.figure(figsize=(15, 5))

    # Fully connected layer heatmap
    plt.subplot(1, len(conv_weights) + 1, 1)

    sns.heatmap(fc_weights, cmap='viridis')
    plt.title('Fully Connected Layer Weights')
    fc_mean = np.mean(fc_weights[:, :256])
    fc_max = np.amax(fc_weights[:, :256])
    fc_min = np.amin(fc_weights[:, :256])
    print(f"fc_mean = {fc_mean}")
    print(f"fc_max = {fc_max}")
    print(f"fc_min = {fc_min}")
    
    # plt.subplot(1, len(conv_weights) + 2, 2)
    # sns.heatmap(fc_weights[:, 256:], cmap='viridis')
    # plt.title('Fully Connected Layer Weights (label embedding)')
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
    plt.savefig(ROOT_DIR / "plots" / "model_weights.png")


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
                    kernel_size=args["gaussian_smoothing_kernel_size"],
                    sigma=args["gaussian_smoothing_sigma"],
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
        
    # create synthetic training set
    datasets = []
    n_samples = int(args["generative_model_n_samples_train"] / len(gen_models))
    for model in gen_models:
        ds = model.create_synthetic_phoneme_dataset(
            n_samples=n_samples,
            neural_window_shape=(1, 256, 32),
        )
        datasets.append(ds)

    train_ds_syn = SyntheticPhonemeDataset.combine_datasets(datasets=datasets)
    assert len(train_ds_syn) == args["generative_model_n_samples_train"]
    assert train_ds_syn.classes == phoneme_classes
    train_dl_syn = DataLoader(
        train_ds_syn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
    )

    # create synthetic val set
    datasets = []
    n_samples = int(args["generative_model_n_samples_val"] / len(gen_models))
    for model in gen_models:
        ds = model.create_synthetic_phoneme_dataset(
            n_samples=n_samples,
            neural_window_shape=(1, 256, 32),
        )
        datasets.append(ds)

    val_ds_syn = SyntheticPhonemeDataset.combine_datasets(datasets=datasets)
    assert len(val_ds_syn) == args["generative_model_n_samples_val"]
    assert val_ds_syn.classes == phoneme_classes
    val_dl_syn = DataLoader(
        val_ds_syn,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
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

    n_classes = len(phoneme_classes)
    args["n_classes"] = n_classes
    model = PhonemeClassifier(n_classes=n_classes, input_shape=args["input_shape"]).to(device)

    n_epochs = args["n_epochs"]
    lr = args["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if n_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)
    
    print(f"len(train_dl_syn.dataset) = {len(train_dl_syn.dataset)}")
    print(f"len(val_dl_syn.dataset) = {len(val_dl_syn.dataset)}")
    print(f"len(val_dl_real.dataset) = {len(val_dl_real.dataset)}")
    print(f"len(test_dl_real.dataset) = {len(test_dl_real.dataset)}")

    output = train_phoneme_classifier(
        model=model,
        train_dl=train_dl_syn,
        test_dls=[val_dl_syn, val_dl_real],
        n_classes=n_classes,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        out_dir=out_dir,
        n_epochs=n_epochs,
        patience=args["patience"],
        model_classes=phoneme_classes,
    )


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    args["n_epochs"] = 10

    for lr in [1e-3]:
        for batch_size in [64]:
            for cls_weights in ["sqrt", None]:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")

                args["output_dir"] = (
                    f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_{batch_size}__lr_{lr}__cls_ws_{cls_weights}__train_on_only_synthetic_{timestamp}"
                )
                # args["generative_model_args_path"] = (
                #     "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_unconditional_20240801_082756/args.json"
                # )
                args["generative_model_weights_path"] = [
                    # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103730/modelWeights",  # cls 3
                    # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103916/modelWeights",  # cls 31

                    # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151151/modelWeights",  # cls 3
                    # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151204/modelWeights",  # cls 31
                    
                    # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120",  # cls 3
                    # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_180210/modelWeights_epoch_120",  # cls 31

                    f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110",  # cls [3, 31]

                ]
                args["generative_model_n_samples_train"] = 20_000
                args["generative_model_n_samples_val"] = 2_000
                print(args["generative_model_weights_path"])

                args["input_shape"] = (128, 8, 8)
                args["lr"] = lr
                args["batch_size"] = batch_size
                args["class_weights"] = cls_weights
                args["transform"] = "softsign"
                args["patience"] = 20
                args["gaussian_smoothing_kernel_size"] = 20
                args["gaussian_smoothing_sigma"] = 2.0
                args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                args["correctness_value"] = ["C"]

                print(
                    "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
                )

                main(args)
