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
    print(f"train_ds_syn.classes = {train_ds_syn.classes}")
    print(f"phoneme_classes = {phoneme_classes}")
    assert train_ds_syn.classes == phoneme_classes
    train_dl_syn = DataLoader(
        train_ds_syn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
    )
    train_dl_syn.name = "train-syn"

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
    val_dl_syn.name = "val-syn"

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

    n_classes = len(phoneme_classes)
    args["n_classes"] = n_classes
    model = PhonemeClassifier(classes=phoneme_classes, input_shape=args["input_shape"]).to(device)

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
        val_dls=[val_dl_syn, val_dl_real],
        test_dl=test_dl_real,
        n_classes=n_classes,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        out_dir=out_dir,
        n_epochs=n_epochs,
        patience=args["patience"],
        model_classes=phoneme_classes,
    )
    print(f"output.keys() = {output.keys()}")


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    # args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")
    args["n_epochs"] = 100

    vae_exp_latent_dim = True

    if vae_exp_latent_dim:

        model_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment")
        lat_dim2epoch = {32: 190, 64: 190, 128: 170, 256: 150, 512: 170}
        for latent_dim, epoch in lat_dim2epoch.items():
            args["output_dir"] = (
                f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/__PhonemeClassifier_bs_64__lr_1e-4__train_on_only_synthetic__latent_dimexp"
            )
            
            args["generative_model_weights_path"] = str(model_dir / f"VAE__latent_dim_{latent_dim}_cond_film/modelWeights_epoch_{epoch}")
            args["generative_model_n_samples_train"] = 10_000
            args["generative_model_n_samples_val"] = 2_000
            args["generative_model_n_samples_test"] = 2_000
            print(args["generative_model_weights_path"])

            args["input_shape"] = (128, 8, 8)
            args["lr"] = 1e-4
            args["batch_size"] = 64
            args["transform"] = "softsign"
            args["patience"] = 30
            args["gaussian_smoothing_kernel_size"] = 20
            args["gaussian_smoothing_sigma"] = 2.0
            args["phoneme_cls"] = [3, 31]
            args["correctness_value"] = ["C"]

            print(
                "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
            )

            main(args)


    # # model_path = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE__latent_dim_32")
    # model_path_3 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[3]")
    # model_path_31 = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_None__phoneme_cls_[31]")

    for batch_size in [64]:
        for n_train in [10_000]:  # [5_000, 10_000, 20_000, 30_000, 40_000, 60_000, 80_000, 100_000, 120_000]:
            for lr in [1e-4]:  # [1e-3, 1e-4]:
                for model_epoch in [90, 100, 110, 120, 130, 140]:  # [850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, ]:  # [850, 900, 950]:  
                # for gen_path in [file for file in model_path_3.glob('*modelWeights*') if file.is_file()]:
                    # print(f"\ngen_path = {gen_path}")

                    # file_name = gen_path.name
                    # print(f"file_name = {file_name}")

                    # gen_path_3 = str(gen_path)
                    # gen_path_31 = str(model_path_31 / file_name)
                    # print(f"gen_path_3 = {gen_path_3}")
                    # print(f"gen_path_31 = {gen_path_31}")
                    
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
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_latent_dim_experiment/VAE_latent_dim_512/modelWeights_epoch_{model_epoch}"

                        
                        # # VAE experiment conditioning
                        f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_experiment_conditioning/VAE__conditioning_concat__phoneme_cls_3_31__dec_emb_dim_8__dec_hidden_dim_256/modelWeights_epoch_{model_epoch}"  # cond experiment


                        # GAN
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173938__phoneme_cls_3_31/modelWeights_1000"  # GAN classes 3, 31
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240818_173804__phoneme_cls_3_31/modelWeights_{model_epoch}"
                        # f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_110045__phoneme_cls_3_31/modelWeights_{model_epoch}"

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
                    args["patience"] = 30
                    args["gaussian_smoothing_kernel_size"] = 20
                    args["gaussian_smoothing_sigma"] = 2.0
                    args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                    args["correctness_value"] = ["C"]

                    print(
                        "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
                    )

                    main(args)
