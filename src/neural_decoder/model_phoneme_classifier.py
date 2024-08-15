import json
from typing import List, Tuple

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
from dataclasses import dataclass

from neural_decoder.phoneme_utils import PHONE_DEF, ROOT_DIR
from text2brain.models.phoneme_image_gan import _get_indices_in_classes

# # 32 * 16 * 16
# class PhonemeClassifier(nn.Module):
#     def __init__(self, n_classes: int):
#         super(PhonemeClassifier, self).__init__()
#         # self.model =
#         self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(256 * 2 * 2, 512)
#         self.fc2 = nn.Linear(512, n_classes)

#     def forward(self, x):
#         # output = self.model(X)
#         # return oputput
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 256 * 2 * 2)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class PhonemeClassifier(nn.Module):
    def __init__(self, classes: List[int], input_shape: Tuple[int, ...]):
        super(PhonemeClassifier, self).__init__()
        self.input_shape = input_shape
        self.classes = classes
        n_classes = len(classes)

        out_dim = 1 if n_classes == 2 else n_classes

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x_reshaped = x.view(-1, *self.input_shape)
        return self.model(x_reshaped)


class PhonemeClassifierRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, num_layers: int = 1):
        super(PhonemeClassifierRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)  # * 2 for bidirectional

    def forward(self, x):
        # Assuming x is of shape (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(
            x.device
        )  # * 2 for bidirectional

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc(out)
        return out


@dataclass
class Stats:
    loss: float
    acc: float
    class_accuracies: np.ndarray
    auroc_macro: float
    auroc_micro: float
    f1_macro: float
    f1_micro: float
    balanced_acc: float
    all_preds_pt: torch.Tensor
    all_labels_pt: torch.Tensor
    all_preds_np_argmax: np.ndarray


def eval_phoneme_classifier(model, dl, criterion, device):
    n_classes = len(model.classes)
    binary = True if n_classes == 2 else False

    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = np.zeros(n_classes)
    class_total = np.zeros(n_classes)

    with torch.no_grad():
        for batch in dl:
            X, y, logits, dayIdx = batch
            y = _get_indices_in_classes(y, torch.tensor(model.classes, device=y.device)).to(y.device)

            X, y, logits, dayIdx = X.to(device), y.to(device), logits.to(device), dayIdx.to(device)

            pred = model(X)
        
            if binary:
                curr_loss = criterion(pred, y.view(y.size(0), 1).float())
                probs = torch.sigmoid(pred)
                pred_labels = (probs > 0.5).float()
            else:
                curr_loss = criterion(pred, y.long())
                pred_labels = torch.argmax(pred, dim=1)
                probs = F.softmax(pred, dim=1)
            loss += curr_loss.item()

            total += pred_labels.size(0)
            correct += (pred_labels == y).sum().item()

            all_preds.append(probs.cpu())
            all_labels.append(y.cpu())

            for label, prediction in zip(y, pred_labels):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
                

    acc = correct / total
    class_accuracies = class_correct / class_total

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert to numpy arrays for sklearn
    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()
    if binary:
        all_preds_np_argmax = (all_preds_np > 0.5)
    else:
        all_preds_np_argmax = np.argmax(all_preds_np, axis=1)

    # Calculate the AUROC
    auroc_macro = roc_auc_score(all_labels_np, all_preds_np_argmax, multi_class="ovr", average="macro")
    auroc_micro = roc_auc_score(all_labels_np, all_preds_np_argmax, multi_class="ovr", average="micro")

    # calculate f1 score
    f1_macro = f1_score(all_labels_np, all_preds_np_argmax, average="macro")
    f1_micro = f1_score(all_labels_np, all_preds_np_argmax, average="micro")

    # Calculate Balanced Accuracy
    balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np_argmax)

    return Stats(
        loss=loss,
        acc=acc,
        class_accuracies=class_accuracies,
        auroc_macro=auroc_macro,
        auroc_micro=auroc_micro,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        balanced_acc=balanced_acc,
        all_labels_pt=all_labels,
        all_preds_pt=all_preds,
        all_preds_np_argmax=all_preds_np_argmax
    )


def train_phoneme_classifier(
    model: PhonemeClassifier,
    train_dl: DataLoader,
    val_dls: List[DataLoader],
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
    stop_training = False
    best_val_acc = 0.0
    count_patience = 0
    all_train_losses = []
    all_val_losses = {dl.name: [] for dl in val_dls}
    all_val_aurocs_macro = {dl.name: [] for dl in val_dls}
    all_val_aurocs_micro = {dl.name: [] for dl in val_dls}
    all_val_f1_macro = {dl.name: [] for dl in val_dls}
    all_val_f1_micro = {dl.name: [] for dl in val_dls}
    all_val_balanced_acc = {dl.name: [] for dl in val_dls}
    time_steps = []
    binary = True if n_classes == 2 else False


    for i_epoch in range(n_epochs):
        for j_batch, data in enumerate(train_dl):
            model.train()
            optimizer.zero_grad()

            X, y, logits, dayIdx = data  # X.size = (batch_size, 1, 32, 256), model will take of reshaping
            y = _get_indices_in_classes(y, torch.tensor(model_classes, device=y.device)).to(y.device)
            X, y, logits, dayIdx = (X.to(device), y.to(device), logits.to(device), dayIdx.to(device))

            pred = model(X)
            
            if binary:
                y = y.view(y.size(0), 1)
                loss = criterion(pred, y.float())
            else:
                loss = criterion(pred, y.long())

            loss.backward()
            optimizer.step()
            all_train_losses.append(loss.item() / y.size(0))

            # evaluate
            if j_batch % 100 == 0:
                # print("\nEval ...")
                time_steps.append({"epoch": i_epoch, "batch": j_batch})

                for val_dl in val_dls:
                    curr_stats = eval_phoneme_classifier(model, val_dl, criterion, device)

                    val_loss = curr_stats.loss
                    all_val_losses[val_dl.name].append(val_loss)

                    val_acc = curr_stats.acc
                    class_accuracies = curr_stats.class_accuracies

                    all_labels = curr_stats.all_labels_pt
                    all_preds = curr_stats.all_preds_pt
                    all_preds_np_argmax = curr_stats.all_preds_np_argmax

                    # auroc
                    val_auroc_macro = curr_stats.auroc_macro
                    all_val_aurocs_macro[val_dl.name].append(val_auroc_macro)

                    val_dl_auroc_micro = curr_stats.auroc_micro
                    all_val_aurocs_micro[val_dl.name].append(val_dl_auroc_micro)

                    # f1 score
                    val_f1_macro = curr_stats.f1_macro
                    val_f1_micro = curr_stats.f1_micro
                    all_val_f1_macro[val_dl.name].append(val_f1_macro)
                    all_val_f1_micro[val_dl.name].append(val_f1_micro)

                    # balanced cccuracy
                    val_balanced_acc = curr_stats.balanced_acc
                    all_val_balanced_acc[val_dl.name].append(val_balanced_acc)

                    print(
                        f"dl: {val_dl.name},\t epoch: {i_epoch}, batch: {j_batch}, val AUROC macro: {val_auroc_macro:.4f}, val AUROC micro: {val_dl_auroc_micro:.4f}, val accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}"
                    )
                    
                    stats = {
                        "time_steps": time_steps,
                        "train_losses": all_train_losses,
                        "val_losses": all_val_losses,
                        "val_aurocs_micro": all_val_aurocs_micro,
                        "val_aurocs_macro": all_val_aurocs_macro,
                        "val_f1_micro": all_val_f1_micro,
                        "val_f1_macro": all_val_f1_macro,
                        "val_balanced_acc": all_val_balanced_acc,
                    }
                    if "real" in val_dl.name:
                        stats["best_val_acc"] =  best_val_acc,
                    
                    for n, acc in enumerate(class_accuracies):
                        stats[f"val_acc_for_phoneme_{n}_{PHONE_DEF[n]}"] = acc

                    with open(out_dir / f"trainingStats.json", "w") as file:
                        json.dump(stats, file, indent=4)

                    if "real" in val_dl.name:
                        if val_acc > best_val_acc:
                            if test_dl is not None:
                                test_stats = eval_phoneme_classifier(model, test_dl, criterion, device)
                                test_acc = test_stats.acc
                                print(f"Test accuracies at the time of best validation acc: test_acc={test_acc}")

                            count_patience = 0
                            model_file = out_dir / "modelWeights"
                            
                            torch.save(model.state_dict(), model_file)
                            best_val_acc = val_acc
                            print(f"New best val accuracy: {best_val_acc:.4f}")

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
                            plt.title(
                                f"Confusion Matrix - Phoneme Classifier (epoch: {i_epoch}, batch: {j_batch})"
                            )
                            plt.savefig(ROOT_DIR / "plots" / "phoneme_classification_heatmap.png")
                            plt.savefig(out_dir / "phoneme_classification_confusion_matrix.png")
                            plt.close()
                        else:
                            count_patience += 1
                            if count_patience == patience:
                                stop_training = True
                                print(f"Stop training due to no improvements after {patience} steps ...")
                                break

                print(f"Patience counter: {count_patience} out of {patience}")

            if stop_training:
                break

        if stop_training:
            break

    return stats
