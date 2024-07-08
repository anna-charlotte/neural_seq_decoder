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
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from neural_decoder.dataloader import MergedDataLoader
from neural_decoder.dataset import PhonemeDataset
from neural_decoder.model_phoneme_classifier import PhonemeClassifier
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from neural_decoder.transforms import SoftsignTransform
from text2brain.models import PhonemeImageGAN
from text2brain.models.phoneme_image_gan import _get_indices_in_classes
from utils import set_seeds


def train_model(
    model: PhonemeClassifier,
    train_dl: DataLoader,
    test_dl: DataLoader,
    n_classes: int,
    optimizer: optim,
    criterion,
    device,
    out_dir,
    n_epochs: int,
    patience: int,
    model_classes: list,
) -> dict:
    print("Train PhonemeClassifier model ...")
    best_test_acc = 0.0
    count_patience = 0
    all_train_losses = []
    all_test_losses = []
    all_test_aurocs_macro = []
    all_test_aurocs_micro = []
    time_steps = []

    for i_epoch in range(n_epochs):
        for j_batch, data in enumerate(train_dl):
            model.train()

            X, y, logits, dayIdx = data
            y = _get_indices_in_classes(y, torch.tensor(model_classes, device=y.device)).to(y.device)

            X, y, logits, dayIdx = (X.to(device), y.to(device), logits.to(device), dayIdx.to(device))
            optimizer.zero_grad()
            # X = X.view(train_dl.batch_size, 32, 16, 16)
            X = X.view(X.size(0), 128, 8, 8)
            pred = model(X)

            loss = criterion(pred, y.long())
            loss.backward()
            optimizer.step()
            all_train_losses.append(loss.item())

            if j_batch > 0 and j_batch % 100 == 0:
                print("Eval ...")
                # evaluate
                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                class_correct = np.zeros(n_classes)
                class_total = np.zeros(n_classes)

                with torch.no_grad():
                    for batch in test_dl:
                        X, y, logits, dayIdx = batch
                        y = _get_indices_in_classes(y, torch.tensor(model_classes, device=y.device)).to(
                            y.device
                        )

                        X, y, logits, dayIdx = (
                            X.to(device),
                            y.to(device),
                            logits.to(device),
                            dayIdx.to(device),
                        )
                        # X = X.view(1, 32, 16, 16)
                        X = X.view(1, 128, 8, 8)

                        pred = model(X)

                        probs = F.softmax(pred, dim=1)

                        loss = criterion(pred, y.long())
                        test_loss += loss.item()

                        pred_labels = torch.argmax(pred, dim=1)
                        total += pred_labels.size(0)
                        correct += (pred_labels == y).sum().item()

                        all_preds.append(probs.cpu())
                        all_labels.append(y.cpu())

                        for label, prediction in zip(y, pred_labels):
                            if label == prediction:
                                class_correct[label] += 1
                            class_total[label] += 1

                all_test_losses.append(test_loss)

                test_acc = correct / total
                class_accuracies = class_correct / class_total
                for n, acc in enumerate(class_accuracies):
                    print(f'Test accuracy for phoneme class {n} ("{PHONE_DEF[n]}"): \t{acc:.4f}')

                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Convert to numpy arrays for sklearn
                all_preds_np = all_preds.numpy()
                all_labels_np = all_labels.numpy()
                all_preds_np_argmax = np.argmax(all_preds_np, axis=1)

                # Calculate the AUROC
                test_auroc_macro = roc_auc_score(
                    all_labels_np, all_preds_np, multi_class="ovr", average="macro"
                )
                all_test_aurocs_macro.append(test_auroc_macro)

                test_auroc_micro = roc_auc_score(
                    all_labels_np, all_preds_np, multi_class="ovr", average="micro"
                )
                all_test_aurocs_micro.append(test_auroc_micro)
                print(
                    f"Epoch: {i_epoch}, batch: {j_batch}, test AUROC macro: {test_auroc_macro:.4f}, test AUROC micro: {test_auroc_micro:.4f}, test accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}"
                )
                time_steps.append({"epoch": i_epoch, "batch": j_batch})
                stats = {
                    "best_test_acc": best_test_acc,
                    "train_losses": all_train_losses,
                    "test_losses": all_test_losses,
                    "test_aurocs_micro": all_test_aurocs_micro,
                    "test_aurocs_macro": all_test_aurocs_macro,
                    "time_steps": time_steps,
                }
                for n, acc in enumerate(class_accuracies):
                    stats[f"test_acc_for_phoneme_{n}_{PHONE_DEF[n]}"] = acc

                with open(out_dir / "trainingStats.json", "w") as file:
                    json.dump(stats, file, indent=4)

                if test_acc > best_test_acc:
                    count_patience = 0
                    model_file = out_dir / "modelWeights"
                    print(f"Saving model checkpoint to: {model_file}")
                    torch.save(model.state_dict(), model_file)
                    best_test_acc = test_acc
                    print(f"New best test accuracy: {best_test_acc:.4f}")

                    # Save the predictions and true labels
                    torch.save(all_preds, out_dir / "all_preds.pt")
                    torch.save(all_labels, out_dir / "all_labels.pt")

                    # Compute and plot the confusion matrix
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
                    plt.title(f"Confusion Matrix - Phoneme Classifier (epoch: {i_epoch}, batch: {j_batch})")
                    plt.savefig(ROOT_DIR / "plots" / "phoneme_classification_heatmap.png")
                    plt.savefig(out_dir / "phoneme_classification_confusion_matrix.png")
                    plt.close()
                else:
                    count_patience += 1
                    if count_patience == patience:
                        break

                print(f"Patience counter: {count_patience} out of {patience}")
    return stats


def get_label_distribution(dataset):
    label_counts = Counter()

    for sample in dataset:
        labels = sample[1].item()
        label_counts.update([labels])

    return label_counts


def plot_phoneme_distribution(
    class_counts,
    out_file: Path,
    title: str = "Phoneme Class Counts",
    x_label: str = "Phoneme",
    y_label: str = "Count",
) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=PHONE_DEF, y=class_counts, palette="muted")  # color=(80 / 255, 80 / 255, 200 / 255, 0.6))

    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_file)


def main(args: dict) -> None:
    for k, v in args.items():
        print(f"{k}: {v}")

    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args["seed"])
    # torch.use_deterministic_algorithms(True)

    batch_size = args["batch_size"]
    device = args["device"]

    train_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    with open(train_file, "rb") as handle:
        data = pickle.load(handle)

    # fmt: off
    class_counts = [
        4841, 7058, 27635, 3298, 2566, 7524, 4674, 2062, 11389, 9501,
        6125, 6573, 4027, 4259, 3315, 5505, 15591, 10434, 2194, 9755,
        13949, 9138, 18297, 3411, 3658, 1661, 6034, 11435, 11605, 2815,
        23188, 2083, 1688, 8414, 6566, 6633, 3707, 7403, 7807
    ]  
    # fmt: on
    plot_phoneme_distribution(
        class_counts,
        ROOT_DIR / "plots" / "phoneme_distribution_training_set_correctly_classified_by_RNN.png",
        "Phoneme Distribution in Training Set",
    )

    # Calculate weights for each class
    if args["class_weights"] == "sqrt":
        class_weights = 1.0 / np.sqrt(np.array(class_counts))
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    elif args["class_weights"] == "inv":
        class_weights = 1.0 / np.array(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None

    phoneme_ds_filter = {"correctness_value": ["C"], "phoneme_cls": list(range(1, 40))}
    args["phoneme_ds_filter"] = phoneme_ds_filter
    phoneme_classes = phoneme_ds_filter["phoneme_cls"]

    transform = None
    if args["transform"] == "softsign":
        transform = SoftsignTransform()
    print(f"transform = {transform.__class__.__name__}")

    train_dl_real = get_data_loader(
        data=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=class_weights,
        transform=transform,
    )
    labels_train = get_label_distribution(train_dl_real.dataset)

    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        data = pickle.load(handle)

    test_dl = get_data_loader(
        data=data,
        batch_size=1,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=None,
        transform=transform,
    )
    labels_test = get_label_distribution(test_dl.dataset)
    class_counts_test = [labels_test[i] for i in range(len(PHONE_DEF))]
    plot_phoneme_distribution(
        class_counts_test,
        ROOT_DIR / "plots" / "phoneme_distribution_test_set_correctly_classified_by_RNN.png",
        "Phoneme Distribution in Test Set",
    )

    print(f"\nlabels_train = {sorted(labels_train)}")
    print(f"\nlabels_test = {sorted(labels_test)}")

    if (
        "generative_model_args_path" in args.keys()
        and "generative_model_weights_path" in args.keys()
        and "generative_model_n_samples" in args.keys()
    ):
        print("Use real and synthetic data ...")

        gen_model = PhonemeImageGAN.load_model(
            args_path=args["generative_model_args_path"],
            weights_path=args["generative_model_weights_path"],
        )

        # neural_window_shape = next(iter(train_dl_real))[0].size()
        n_synthetic_samples = args["generative_model_n_samples"]
        synthetic_ds = gen_model.create_synthetic_phoneme_dataset(
            n_samples=n_synthetic_samples,
            neural_window_shape=(32, 256),
        )
        synthetic_dl = DataLoader(
            synthetic_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=None,
        )

        train_dl = MergedDataLoader(train_dl_real, synthetic_dl)
    else:
        print("Use only real data ...")
        train_dl = train_dl_real

    n_classes = len(phoneme_classes)
    args["n_classes"] = n_classes
    model = PhonemeClassifier(n_classes=n_classes).to(device)

    n_epochs = args["n_epochs"]
    lr = args["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)
    with open(out_dir / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    output = train_model(
        model=model,
        train_dl=train_dl,
        test_dl=test_dl,
        n_classes=n_classes,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        out_dir=out_dir,
        n_epochs=n_epochs,
        patience=10,
        model_classes=phoneme_classes,
    )

    best_auroc = output["best_auroc"]


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["train_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    )
    args["test_set_path"] = (
        "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    )
    args["n_epochs"] = 10

    for lr in [1e-4]:
        for batch_size in [64, 128]:
            for cls_weights in ["sqrt", None]:
                for synthetic_data in [False, True]:

                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")

                    args["output_dir"] = (
                        f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_{batch_size}__lr_{lr}__cls_ws_{cls_weights}__synthetic_{synthetic_data}_{timestamp}"
                    )
                    if synthetic_data:
                        args["generative_model_args_path"] = (
                            "/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_20240707_132459/args"
                        )
                        args["generative_model_weights_path"] = (
                            "/data/engs-pnpl/lina4471/willett2023/generative_models/PhonemeImageGAN_20240707_132459/modelWeights_epoch_1"
                        )
                        args["generative_model_n_samples"] = 50_000
                        print(args["generative_model_weights_path"])
                    args["lr"] = lr
                    args["batch_size"] = batch_size
                    args["class_weights"] = cls_weights
                    args["transform"] = "softsign"

                    # args["n_input_features"] = 41
                    # args["n_output_features"] = 256
                    # args["hidden_dim"] = 512
                    # args["n_layers"] = 2

                    if "generative_model_weights_path" in args.keys():
                        print("\nTrain phoeneme classifier using REAL and SYNTHETIC data!")
                    else:
                        print("\nTrain phoeneme classifier using only REAL data!")
                    main(args)
