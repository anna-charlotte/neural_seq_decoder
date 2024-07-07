import pickle
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from neural_decoder.dataset import PhonemeDataset
from neural_decoder.model_phoneme_classifier import PhonemeClassifier
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR


def plot_accuracies(
    accs,
    out_file: Path,
    title: str = "Phoneme Classification - Test Accuracies (Baseline Classifier)",
    x_label: str = "Phoneme",
    y_label: str = "Test Accuracy",
) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=PHONE_DEF, y=accs, palette="muted")

    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_file)


def main() -> None:

    model_dir = Path(
        "/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_64_lr_0.0001"
    )

    args_file = model_dir / "args"
    print(f"args_file = {args_file}")
    with open(args_file, "rb") as file:
        args = pickle.load(file)

    # args = {}
    # args["seed"] = 0
    # args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # args[
    #     "train_set_path"
    # ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    # args[
    #     "test_set_path"
    # ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    # args["n_epochs"] = 10
    # args["lr"] = 1e-4
    # args["batch_size"] = 64
    # args["class_weights"] = "sqrt"

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    batch_size = args["batch_size"]
    device = args["device"]

    # train_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    # with open(train_file, "rb") as handle:
    #     data = pickle.load(handle)

    # # fmt: off
    # class_counts = [
    #     4841, 7058, 27635, 3298, 2566, 7524, 4674, 2062, 11389, 9501,
    #     6125, 6573, 4027, 4259, 3315, 5505, 15591, 10434, 2194, 9755,
    #     13949, 9138, 18297, 3411, 3658, 1661, 6034, 11435, 11605, 2815,
    #     23188, 2083, 1688, 8414, 6566, 6633, 3707, 7403, 7807
    # ]
    # # fmt: on

    # # Calculate weights for each class
    # if args["class_weights"] == "sqrt":
    #     class_weights = 1.0 / np.sqrt(np.array(class_counts))
    #     class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # elif args["class_weights"] == "inv":
    #     class_weights = 1.0 / np.array(class_counts)
    #     class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # else:
    #     class_weights = None

    # train_dl = get_data_loader(
    #     data=data,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=None,
    #     dataset_cls=PhonemeDataset,
    #     phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": list(range(1, 40))},
    #     class_weights=class_weights,
    # )

    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        data = pickle.load(handle)

    test_dl = get_data_loader(
        data=data,
        batch_size=1,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": list(range(1, 40))},
        class_weights=None,
    )

    n_classes = 39
    model = PhonemeClassifier(n_classes=n_classes).to(device)
    model_weight_path = model_dir / "modelWeights"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = np.zeros(n_classes)
    class_total = np.zeros(n_classes)

    all_test_aurocs_macro = []
    all_test_aurocs_micro = []

    with torch.no_grad():
        for batch in test_dl:
            X, y, logits, dayIdx = batch
            X, y, logits, dayIdx = (
                X.to(device),
                y.to(device),
                logits.to(device),
                dayIdx.to(device),
            )
            X = X.view(1, 128, 8, 8)
            if n_classes < 41:
                y = y - 1
            pred = model(X)

            probs = F.softmax(pred, dim=1)

            pred_labels = torch.argmax(pred, dim=1)
            total += pred_labels.size(0)
            correct += (pred_labels == y).sum().item()

            all_preds.append(probs.cpu())
            all_labels.append(y.cpu())

            for label, prediction in zip(y, pred_labels):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1

    test_acc = correct / total
    class_accuracies = class_correct / class_total
    for n, acc in enumerate(class_accuracies):
        print(f'Test accuracy for phoneme class {n} ("{PHONE_DEF[n]}"): \t{acc:.4f}')

    plot_accuracies(
        class_accuracies, out_file=(ROOT_DIR / "plots" / f"test_accs_{model.__class__.__name__}.png"),
    )

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert to numpy arrays for sklearn
    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()
    all_preds_np_argmax = np.argmax(all_preds_np, axis=1)

    # Calculate the AUROC
    test_auroc_macro = roc_auc_score(all_labels_np, all_preds_np, multi_class="ovr", average="macro")
    all_test_aurocs_macro.append(test_auroc_macro)

    test_auroc_micro = roc_auc_score(all_labels_np, all_preds_np, multi_class="ovr", average="micro")
    all_test_aurocs_micro.append(test_auroc_micro)
    print(
        f"Test AUROC macro: {test_auroc_macro:.4f}, test AUROC micro: {test_auroc_micro:.4f}, test accuracy: {test_acc:.4f}"
    )

    # Compute and plot the confusion matrix
    cm = confusion_matrix(all_labels, all_preds_np_argmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=PHONE_DEF, yticklabels=PHONE_DEF,
    )
    plt.xlabel("Predicted Phoneme")
    plt.ylabel("True Phoneme")
    plt.title(f"Confusion Matrix - Phoneme Classifier (CNN)")
    plt.savefig(ROOT_DIR / "plots" / "testtest_.png")
    plt.close()

    # fig, ax = plt.subplots(figsize=(12, 8))

    # # Create the heatmap
    # sns.heatmap(
    #     cm,
    #     annot=True,
    #     fmt="d",
    #     cmap="Blues",
    #     xticklabels=PHONE_DEF,
    #     yticklabels=PHONE_DEF,
    #     ax=ax,
    #     cbar=True
    # )

    # ax.set_xlabel("Predicted Phoneme")
    # ax.set_ylabel("True Phoneme")
    # ax.set_title("Phoneme classification - confusion matrix ")

    # # Create secondary x-axis for accuracies
    # ax2 = ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(np.arange(len(PHONE_DEF)) + 0.5)
    # ax2.set_xticklabels([f'{acc:.3f}' for acc in class_accuracies], rotation=90)

    # ax2.set_xlabel("Class Accuracy")

    # # Remove the numbers in the first row of the heatmap
    # for t in ax.texts:
    #     if int(t.get_text()) == cm[0, int(t.get_position()[1] - 0.5)]:
    #         t.set_text("")

    # plt.tight_layout()
    # plt.savefig(ROOT_DIR / "plots" / "testtest.png")
    # plt.close()


if __name__ == "__main__":
    main()
