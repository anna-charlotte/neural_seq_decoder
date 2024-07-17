import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy
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

from neural_decoder.data.dataset import PhonemeDataset
from neural_decoder.model_phoneme_classifier import PhonemeClassifier
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from text2brain.visualization import plot_accuracies, plot_metric_of_multiple_models
from utils import set_seeds


def main() -> None:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    prefix = f"phoneme_classification_{timestamp}"

    model_file_names = [
        "PhonemeClassifier_bs_64__lr_0.0001__cls_ws_sqrt__synthetic_True",
        "PhonemeClassifier_bs_64__lr_0.0001__cls_ws_sqrt__synthetic_False",
    ]

    model2metrics = {}

    for i, model_file_name in enumerate(model_file_names):
        print(f"\nLoading and evaluating model number {i+1}")

        model_dir = (
            Path(
                # "/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_64_lr_0.0001"
                "/data/engs-pnpl/lina4471/willett2023/phoneme_classifier"
            )
            / model_file_name
        )

        args_file = model_dir / "args"
        print(f"args_file = {args_file}")
        with open(args_file, "rb") as file:
            args = pickle.load(file)

        for k, v in args.items():
            print(f"{k}: {v}")

        set_seeds(args["seed"])

        device = args["device"]
        phoneme_ds_filter = (
            {"correctness_value": ["C"], "phoneme_cls": list(range(1, 40))}
            if not "phoneme_ds_filter" in args.keys()
            else args["phoneme_ds_filter"]
        )

        test_data_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
        with open(test_data_file, "rb") as handle:
            data = pickle.load(handle)

        test_dl = get_data_loader(
            data=data,
            batch_size=1,
            shuffle=False,
            collate_fn=None,
            dataset_cls=PhonemeDataset,
            phoneme_ds_filter=phoneme_ds_filter,
            class_weights=None,
        )

        n_classes = 39 if not "n_classes" in args.keys() else args["n_classes"]
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
                y = _get_indices_in_classes(
                    y, torch.tensor(phoneme_ds_filter["phoneme_cls"], device=y.device)
                ).to(y.device)
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

        # plot test accuracies over each classes
        plot_accuracies(
            class_accuracies,
            out_file=(ROOT_DIR / "evaluation" / f"{prefix}_test_accs_{model_file_name}.png"),
            title=f"Phoneme Classification - Test Accuracies \n (model: {model_file_name})",
        )

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_preds_np = all_preds.numpy()
        all_labels_np = all_labels.numpy()
        all_preds_np_argmax = np.argmax(all_preds_np, axis=1)

        # calculate the AUROCs
        test_auroc_macro = roc_auc_score(all_labels_np, all_preds_np, multi_class="ovr", average="macro")
        all_test_aurocs_macro.append(test_auroc_macro)

        test_auroc_micro = roc_auc_score(all_labels_np, all_preds_np, multi_class="ovr", average="micro")
        all_test_aurocs_micro.append(test_auroc_micro)
        print(
            f"Test AUROC macro: {test_auroc_macro:.4f}, test AUROC micro: {test_auroc_micro:.4f}, test accuracy: {test_acc:.4f}"
        )

        # calculate f1 score
        test_f1_macro = f1_score(all_labels_np, all_preds_np_argmax, average="macro")
        test_f1_micro = f1_score(all_labels_np, all_preds_np_argmax, average="micro")

        # Calculate Balanced Accuracy
        test_balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np_argmax)

        metrics = {}
        metrics["best_test_acc_overall"] = test_acc
        metrics["class_accuracies"] = class_accuracies.tolist()
        metrics["test_auroc_micro"] = test_auroc_micro
        metrics["test_auroc_macro"] = test_auroc_macro
        metrics["test_f1_micro"] = test_f1_micro
        metrics["test_f1_macro"] = test_f1_macro
        metrics["test_balanced_acc"] = test_balanced_acc

        for n, acc in enumerate(class_accuracies):
            metrics[f"test_acc_for_phoneme_{n}_{PHONE_DEF[n]}"] = acc

        model2metrics[model_file_name] = metrics

        # compute and plot the confusion matrix
        cm = confusion_matrix(all_labels, all_preds_np_argmax)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=PHONE_DEF,
            yticklabels=PHONE_DEF,
        )
        plt.xlabel("Predicted Phoneme")
        plt.ylabel("True Phoneme")
        plt.title(f"Confusion Matrix - Phoneme Classifier (CNN) \n (model: {model_file_name})")
        plt.savefig(ROOT_DIR / "evaluation" / f"phoneme_classification_{timestamp}_CM_{model_file_name}.png")
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
        # plt.savefig(ROOT_DIR / "evaluation" / "testtest.png")
        # plt.close()

    model_names = list(model2metrics.keys())
    accuracies = [model2metrics[m]["class_accuracies"] for m in model_names]
    plot_metric_of_multiple_models(
        model_names=model_names,
        values=accuracies,
        classes=PHONE_DEF,
        title="Phoneme Classififers - comparison of accuracies on the test set",
        out_file=ROOT_DIR / "evaluation" / f"{prefix}_model_comparison_test_accuracies.png",
        xlabel="phoneme classes",
        ylabel="accuracy",
    )

    with open(ROOT_DIR / "evaluation" / f"{prefix}_trainingStats.json", "w") as file:
        json.dump(model2metrics, file, indent=4)

    model2accs = {m: model2metrics[m]["class_accuracies"] for m in model2metrics.keys()}
    print(f"model2accs = {model2accs}")

    accs = list(model2accs.values())
    for acc1, acc2 in zip(accs[0], accs[1]):
        stats, p_val = scipy.stats.mannwhitneyu([acc1], [acc2], alternative="two-sided")
        print(f"\nstats = {stats}")
        print(f"p_val = {p_val}")

        stats, p_val = scipy.stats.mannwhitneyu(
            [acc1 * 100, acc1, acc1 * 2], [acc2 * 100, acc2, acc2], alternative="two-sided"
        )
        print(f"stats = {stats}")
        print(f"p_val = {p_val}")

        stats, p_val = scipy.stats.mannwhitneyu([acc1 * 2], [acc2], alternative="two-sided")
        print(f"stats = {stats}")
        print(f"p_val = {p_val}")

        stats, p_val = scipy.stats.mannwhitneyu([acc1 * 1], [acc2 * 10], alternative="two-sided")
        print(f"stats = {stats}")
        print(f"p_val = {p_val}")


if __name__ == "__main__":
    main()
