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

from data.dataset import PhonemeDataset
from neural_decoder.model_phoneme_classifier import (
    PhonemeClassifier,
    train_phoneme_classifier,
)
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models import PhonemeImageGAN
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from utils import set_seeds


def get_label_distribution(dataset):
    label_counts = Counter()

    for sample in dataset:
        labels = sample[1].item()
        label_counts.update([labels])

    return label_counts


def main(args: dict) -> None:
    print("Training with the following arguments:")
    for k, v in args.items():
        print(f"{k}: {v}")

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args["seed"])

    batch_size = args["batch_size"]
    device = args["device"]

    phoneme_ds_filter = {"correctness_value": args["correctness_value"], "phoneme_cls": args["phoneme_cls"]}
    args["phoneme_ds_filter"] = phoneme_ds_filter
    phoneme_classes = args["phoneme_cls"]

    transform = None
    if args["transform"] == "softsign":
        transform = SoftsignTransform()
    print(f"transform = {transform.__class__.__name__}")

    gen_model = PhonemeImageGAN.load_model(
        args_path=args["generative_model_args_path"],
        weights_path=args["generative_model_weights_path"],
    )

    # synthetic training set
    train_ds_syn = gen_model.create_synthetic_phoneme_dataset(
        n_samples=args["generative_model_n_samples"],
        neural_window_shape=(32, 256),
    )
    train_dl_syn = DataLoader(
        train_ds_syn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
    )

    # synthetic test set
    test_ds_syn = gen_model.create_synthetic_phoneme_dataset(
        n_samples=4_000,
        neural_window_shape=(32, 256),
    )
    test_dl_syn = DataLoader(
        test_ds_syn,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=None,
    )

    # real test set
    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        test_data = pickle.load(handle)

    test_dl_real = get_data_loader(
        data=test_data,
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    output = train_phoneme_classifier(
        model=model,
        train_dl=train_dl_syn,
        test_dls=[test_dl_syn, test_dl_real],
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
    args["test_set_path"] = str(data_dir / "rnn_test_set_with_logits.pkl")
    args["n_epochs"] = 10

    for lr in [1e-4]:
        for batch_size in [64, 128]:
            for cls_weights in ["sqrt", None]:
                for synthetic_data in [True]:
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")

                    args["output_dir"] = (
                        f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_{batch_size}__lr_{lr}__cls_ws_{cls_weights}__train_on_only_synthetic_{timestamp}"
                    )
                    if synthetic_data:
                        args["generative_model_args_path"] = (
                            "/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_20240708_103107/args"
                        )
                        args["generative_model_weights_path"] = (
                            "/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_20240708_103107/modelWeights_epoch_6"
                        )
                        args["generative_model_n_samples"] = 50_000
                        print(args["generative_model_weights_path"])

                    args["input_shape"] = (128, 8, 8)
                    args["lr"] = lr
                    args["batch_size"] = batch_size
                    args["class_weights"] = cls_weights
                    args["transform"] = "softsign"
                    args["patience"] = 20
                    args["gaussian_smoothing"] = 2.0
                    args["phoneme_cls"] = list(range(1, 40))
                    args["correctness_value"] = ["C"]

                    print(
                        "\nTrain phoeneme classifier on SYNTHETIC data. Test on SYNTHETIC as well as REAL data."
                    )

                    main(args)
