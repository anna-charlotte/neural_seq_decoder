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
    def __init__(self, n_classes: int, input_shape: Tuple[int, ...]):
        super(PhonemeClassifier, self).__init__()
        self.input_shape = input_shape
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
            nn.Linear(512, n_classes),
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


def train_phoneme_classifier(
    model: PhonemeClassifier,
    train_dl: DataLoader,
    test_dls: List[DataLoader],
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
    best_test_acc = 0.0
    count_patience = 0
    all_train_losses = []
    all_test_losses = []
    all_test_aurocs_macro = []
    all_test_aurocs_micro = []
    all_test_f1_macro = []
    all_test_f1_micro = []
    all_test_balanced_acc = []
    time_steps = []

    for i_epoch in range(n_epochs):
        for j_batch, data in enumerate(train_dl):
            model.train()
            optimizer.zero_grad()

            X, y, logits, dayIdx = data  # X.size = (batch_size, 1, 32, 256), model will take of reshaping
            y = _get_indices_in_classes(y, torch.tensor(model_classes, device=y.device)).to(y.device)
            X, y, logits, dayIdx = (X.to(device), y.to(device), logits.to(device), dayIdx.to(device))

            pred = model(X)

            loss = criterion(pred, y.long())
            loss.backward()
            optimizer.step()
            all_train_losses.append(loss.item())

            # evaluate
            if j_batch > 0 and j_batch % 100 == 0:
                print("\nEval ...")

                for k_test_dl, test_dl in enumerate(test_dls):
                    print(f"k_test_dl = {k_test_dl}")
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

                    if k_test_dl == 0:
                        all_test_losses.append(test_loss)

                    test_acc = correct / total
                    class_accuracies = class_correct / class_total
                    # for n, acc in enumerate(class_accuracies):
                    #     print(f'Test accuracy for phoneme class {n} ("{PHONE_DEF[n]}"): \t{acc:.4f}')

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
                    if k_test_dl == 0:
                        all_test_aurocs_macro.append(test_auroc_macro)

                    test_auroc_micro = roc_auc_score(
                        all_labels_np, all_preds_np, multi_class="ovr", average="micro"
                    )
                    if k_test_dl == 0:
                        all_test_aurocs_micro.append(test_auroc_micro)

                    # calculate f1 score
                    test_f1_macro = f1_score(all_labels_np, all_preds_np_argmax, average="macro")
                    test_f1_micro = f1_score(all_labels_np, all_preds_np_argmax, average="micro")
                    if k_test_dl == 0:
                        all_test_f1_macro.append(test_f1_macro)
                        all_test_f1_micro.append(test_f1_micro)

                    # Calculate Balanced Accuracy
                    test_balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np_argmax)
                    if k_test_dl == 0:
                        all_test_balanced_acc.append(test_balanced_acc)

                    print(
                        f"k_test_dl: {k_test_dl} ({'fake' if k_test_dl == 0 else 'real'}), epoch: {i_epoch}, batch: {j_batch}, test AUROC macro: {test_auroc_macro:.4f}, test AUROC micro: {test_auroc_micro:.4f}, test accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}"
                    )
                    if k_test_dl == 0:
                        time_steps.append({"epoch": i_epoch, "batch": j_batch})
                        stats = {
                            "time_steps": time_steps,
                            "best_test_acc": best_test_acc,
                            "train_losses": all_train_losses,
                            "test_losses": all_test_losses,
                            "test_aurocs_micro": all_test_aurocs_micro,
                            "test_aurocs_macro": all_test_aurocs_macro,
                            "test_f1_micro": all_test_f1_micro,
                            "test_f1_macro": all_test_f1_macro,
                            "test_balanced_acc": all_test_balanced_acc,
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
