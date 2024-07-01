import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR


def softmax(logits):
    logits_tensor = torch.tensor(logits)
    probabilities = torch.nn.functional.softmax(logits_tensor, dim=0)
    return probabilities


def main(args: dict) -> None:

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.use_deterministic_algorithms(True)

    device = args["device"]
    batch_size = args["batch_size"]

    for phoneme_cls in range(41):
        print(f"device = {device}")

        train_file = args["train_set_path"]
        with open(train_file, "rb") as handle:
            train_data = pickle.load(handle)

        train_dl = get_data_loader(
            data=train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=None,
            dataset_cls=PhonemeDataset,
            phoneme_cls=phoneme_cls if isinstance(phoneme_cls, int) else None,
        )
        train_ds = train_dl.dataset
        print(f"len(train_ds) = {len(train_ds)}")
        all_logits_for_phoneme = []
        all_probabilities = []
        for batch in train_ds:
            _, _, logits, _= batch
            probabilities = softmax(logits)
            assert phoneme_cls == torch.argmax(logits).item()
            logit = logits[phoneme_cls].item()
            all_logits_for_phoneme.append(logit)

            prob = probabilities[phoneme_cls]
            # print()
            # print(logit)
            # print(sum(logits))
            # print(prob)
            # print(probabilities)
            all_probabilities.append(prob)

        # print(all_logits_for_phoneme)

        plt.figure(figsize=(10, 6))
        plt.hist(all_logits_for_phoneme, bins=40, edgecolor='black')
        plt.title(f'Distribution of Logit Values (B2T RNN) for Phoneme {phoneme_cls} (total counts = {len(all_logits_for_phoneme)})')
        plt.xlabel('Logit Values')
        plt.ylabel('Frequency')
        plt.xlim(left=0)
        plt.xlim(right=1)
        plt.grid(True)

        plt.savefig(ROOT_DIR / "plots" / "phoneme_distr_logits" / f"logits_phoneme_{phoneme_cls}.png")

        plt.figure(figsize=(10, 6))
        plt.hist(all_probabilities, bins=40, edgecolor='black')
        plt.title(f'Distribution of Probabilitiy Values (B2T RNN) for Phoneme {phoneme_cls} (total counts = {len(all_probabilities)})')
        plt.xlabel('Probabilities Values')
        plt.ylabel('Frequency')
        plt.xlim(left=0)
        plt.xlim(right=1)
        plt.grid(True)

        plt.savefig(ROOT_DIR / "plots" / "phoneme_distr_probabilities" / f"prob_phoneme_{phoneme_cls}.png")


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda"
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    # args["output_dir"] = "/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn"
    args["batch_size"] = 8
    args["n_input_features"] = 41
    args["n_output_features"] = 256
    args["hidden_dim"] = 512
    args["n_layers"] = 2
    main(args)
