import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from data.augmentations import GaussianSmoothing
from data.dataset import PhonemeDataset, SyntheticPhonemeDataset
from data.dataloader import MergedDataLoader
from neural_decoder.model_phoneme_classifier import (
    PhonemeClassifier,
    train_phoneme_classifier,
)
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models import PhonemeImageGAN
from text2brain.models.model_interface_load import load_t2b_gen_model
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from text2brain.visualization import plot_phoneme_distribution
from utils import load_pkl, set_seeds


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
    print(f"phoneme_ds_filter = {phoneme_ds_filter}")

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

    # fmt: off
    class_counts = [
        4841, 7058, 27635, 3298, 2566, 7524, 4674, 2062, 11389, 9501,
        6125, 6573, 4027, 4259, 3315, 5505, 15591, 10434, 2194, 9755,
        13949, 9138, 18297, 3411, 3658, 1661, 6034, 11435, 11605, 2815,
        23188, 2083, 1688, 8414, 6566, 6633, 3707, 7403, 7807
    ] # fmt: on
    if len(phoneme_classes) < len(class_counts):
        class_counts = [class_counts[i-1] for i in phoneme_classes]

    # # plot phoneme distribution
    # plot_phoneme_distribution(
    #     class_counts,
    #     ROOT_DIR / "plots" / "phoneme_distribution_training_set_correctly_classified_by_RNN.png",
    #     f"Phoneme Distribution in Training Set ({sum(class_counts)} samples)",
    # )

    # Calculate weights for each class
    if args["class_weights"] == "sqrt":
        class_weights = torch.tensor(1.0 / np.sqrt(np.array(class_counts))).float()
        # class_weights = torch.tensor(class_weights, dtype=torch.float32)
    elif args["class_weights"] == "inv":
        class_weights = torch.tensor(1.0 / np.array(class_counts)).float()
        # class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    print(f"transform = {transform.__class__.__name__}")

    # load train dataloader
    train_dl_real = get_data_loader(
        data=load_pkl(args["train_set_path"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=class_weights,
        transform=transform,
    )
    train_dl_real.name = "train-real"
    labels_train = get_label_distribution(train_dl_real.dataset)

    # load val dataloader
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
    labels_val = get_label_distribution(val_dl_real.dataset)
    class_counts_val = [labels_val[i] for i in range(1, 40)]
    plot_phoneme_distribution(
        class_counts_val,
        ROOT_DIR / "plots" / "phoneme_distribution_test_set_VAL_SPLIT_correctly_classified_by_RNN.png",
        f"Phoneme Distribution in Validation Set ({len(val_dl_real.dataset)} samples)",
    )

    # load test dataloader
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
    labels_test = get_label_distribution(test_dl_real.dataset)
    class_counts_test = [labels_test[i] for i in range(1, 40)]
    plot_phoneme_distribution(
        class_counts_test,
        ROOT_DIR / "plots" / "phoneme_distribution_test_set_TEST_SPLIT_correctly_classified_by_RNN.png",
        f"Phoneme Distribution in Test Set ({len(test_dl_real.dataset)} samples)",
    )

    if (
       "generative_model_weights_path" in args.keys()
        and "generative_model_n_samples" in args.keys()
    ):
        print("Use real and synthetic data ...")
        if isinstance(args["generative_model_weights_path"], list):
            gen_models = []
            for weights_path in args["generative_model_weights_path"]:
                model = load_t2b_gen_model(weights_path=weights_path)
                gen_models.append(model)
        else:
            gen_models = [load_t2b_gen_model(weights_path=args["generative_model_weights_path"])]

            
        # create synthetic training set
        datasets = []
        n_samples = int(args["generative_model_n_samples"] / len(gen_models))
        for model in gen_models:
            ds = model.create_synthetic_phoneme_dataset(
                n_samples=n_samples,
                neural_window_shape=(1, 256, 32),
            )
            datasets.append(ds) 

        synthetic_ds = SyntheticPhonemeDataset.combine_datasets(datasets=datasets)
        assert len(synthetic_ds) == args["generative_model_n_samples"]
        assert synthetic_ds.classes == phoneme_classes
        synthetic_dl = DataLoader(
            synthetic_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=None,
        )


        # synthetic_ds = gen_model.create_synthetic_phoneme_dataset(
        #     n_samples=args["generative_model_n_samples"],
        #     neural_window_shape=(1, 256, 32),
        # )

        # synthetic_dl = DataLoader(
        #     synthetic_ds,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=0,
        #     pin_memory=True,
        #     collate_fn=None,
        # )

        train_dl = MergedDataLoader(
            loader1=synthetic_dl, loader2=train_dl_real, prop1=args["generative_model_proportion"]
        )
    else:
        print("Use only real data ...")
        train_dl = train_dl_real

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

    # print(f"len(train_dl.dataset) = {len(train_dl.dataset)}")
    # print(f"len(synthetic_dl.dataset) = {len(synthetic_dl.dataset)}")
    print(f"len(val_dl_real.dataset) = {len(val_dl_real.dataset)}")
    print(f"len(test_dl_real.dataset) = {len(test_dl_real.dataset)}")

    output = train_phoneme_classifier(
        model=model,
        train_dl=train_dl,
        val_dls=[val_dl_real],  # test_dl,
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


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")
    args["val_set_path"] = str(data_dir / "rnn_test_set_with_logits_VAL_SPLIT.pkl")
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits_TEST_SPLIT.pkl")

    # args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits.pkl")
    args["n_epochs"] = 100

    for synthetic_data in [False, True]:
        for batch_size in [64]:
            for cls_weights in [None]:
                for lr in [1e-4]:
                    for n_samples_syn in [40_000]:
                        now = datetime.now()
                        timestamp = now.strftime("%Y%m%d_%H%M%S")

                        args["output_dir"] = (
                            f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_{batch_size}__lr_{lr}__cls_ws_{cls_weights}__synthetic_{synthetic_data}_{timestamp}"
                        )
                        if synthetic_data:
                            # args["generative_model_args_path"] = (
                            #     "/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_20240708_103107/args"
                            # )
                            # args["generative_model_weights_path"] = f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110"  # cls [3, 31]
                            args["generative_model_weights_path"] = [
                            #     # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103730/modelWeights",  # cls 3
                            #     # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs/VAE_conditional_20240807_103916/modelWeights",  # cls 31

                            #     # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151151/modelWeights",  # cls 3
                            #     # "/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_151204/modelWeights",  # cls 31
                                
                            #     # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_182747/modelWeights_epoch_120",  # cls 3
                            #     # f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_conditional_20240807_180210/modelWeights_epoch_120",  # cls 31

                            #     f"/data/engs-pnpl/lina4471/willett2023/generative_models/VAEs_binary/VAE_unconditional_20240809_044252/modelWeights_epoch_110",  # cls [3, 31]

                                f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_120647__phoneme_cls_3/modelWeights_1150",
                                f"/data/engs-pnpl/lina4471/willett2023/generative_models/GANs/test__PhonemeImageGAN_20240819_235138__phoneme_cls_31/modelWeights_1150"

                            ]

                            args["generative_model_n_samples"] = n_samples_syn
                            args["generative_model_proportion"] = None
                            print(args["generative_model_weights_path"])

                        args["input_shape"] = (128, 8, 8)
                        args["lr"] = lr
                        args["batch_size"] = batch_size
                        args["class_weights"] = cls_weights
                        args["transform"] = "softsign"
                        args["patience"] = 60
                        args["gaussian_smoothing_kernel_size"] = 20
                        args["gaussian_smoothing_sigma"] = 2.0
                        args["phoneme_cls"] = [3, 31]  # list(range(1, 40))
                        args["correctness_value"] = ["C"]

                        args["unconditional"] = True

                        # args["n_input_features"] = 41
                        # args["n_output_features"] = 256
                        # args["hidden_dim"] = 512
                        # args["n_layers"] = 2

                        if "generative_model_weights_path" in args.keys():
                            print("\nTrain phoeneme classifier using REAL and SYNTHETIC data!")
                        else:
                            print("\nTrain phoeneme classifier using only REAL data!")
                        main(args)
