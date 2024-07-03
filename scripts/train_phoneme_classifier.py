import pickle
import random
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
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.phoneme_utils import ROOT_DIR

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


# 128 * 8 * 8
# class PhonemeClassifier(nn.Module):
#     def __init__(self, n_classes: int):
#         super(PhonemeClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(256 * 1 * 1, 512)
#         self.fc2 = nn.Linear(512, n_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 256 * 1 * 1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
class PhonemeClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(PhonemeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    n_classes: int,
    optimizer: optim,
    criterion,
    device,
    out_dir,
    n_epochs: int,
    patience: int,
    auroc_average: str = "macro",
):
    best_auroc = 0.0
    count_patience = 0
    all_train_losses = []
    all_test_losses = []
    all_test_aurocs = []

    for i in range(n_epochs):
        for j, data in enumerate(train_dl):
            model.train()

            X, y, logits, dayIdx = data
            if n_classes < 41:
                y = y - 1

            X, y, logits, dayIdx = (X.to(device), y.to(device), logits.to(device), dayIdx.to(device))
            optimizer.zero_grad()
            # X = X.view(train_dl.batch_size, 32, 16, 16)
            X = X.view(train_dl.batch_size, 128, 8, 8)
            pred = model(X)

            loss = criterion(pred, y.long())
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                # evaluate
                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for batch in test_dl:
                        X, y, logits, dayIdx = batch
                        X, y, logits, dayIdx = (
                            X.to(device),
                            y.to(device),
                            logits.to(device),
                            dayIdx.to(device),
                        )
                        # X = X.view(1, 32, 16, 16)
                        X = X.view(1, 128, 8, 8)
                        if n_classes < 41:
                            y = y - 1
                        pred = model(X)

                        probs = F.softmax(pred, dim=1)

                        loss = criterion(pred, y.long())
                        test_loss += loss.item()

                        pred_labels = torch.argmax(pred, dim=1)
                        total += pred_labels.size(0)
                        correct += (pred_labels == y).sum().item()

                        all_preds.append(probs.cpu())
                        all_labels.append(y.cpu())

                test_acc = correct / total

                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # Convert to numpy arrays for sklearn
                all_preds_np = all_preds.numpy()
                all_labels_np = all_labels.numpy()
                all_preds_np_argmax = np.argmax(all_preds_np, axis=1)

                # Calculate the AUROC
                test_auroc = roc_auc_score(
                    all_labels_np, all_preds_np, multi_class="ovr", average=auroc_average
                )
                print(
                    f"Epoch: {i}, batch: {j}, test AUROC: {test_auroc:.4f}, test accuracy: {test_acc:.4f}, test_loss: {test_loss:.4f}"
                )

                if test_auroc > best_auroc:
                    count_patience = 0
                    torch.save(model.state_dict(), out_dir / "modelWeights")
                    best_auroc = test_auroc

                    # Compute the confusion matrix
                    cm = confusion_matrix(all_labels, all_preds_np_argmax)

                    # Plot the heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=range(n_classes),
                        yticklabels=range(n_classes),
                    )
                    plt.xlabel("Predicted Phoneme")
                    plt.ylabel("True Phoneme")
                    plt.title("Confusion Matrix - Phoneme Classifier")
                    plt.savefig(ROOT_DIR / "plots" / "phoneme_classification_heatmap.png")
                    plt.savefig(out_dir / "phoneme_classification_heatmap.png")
                    plt.close()
                else:
                    count_patience += 1
                    if count_patience == patience:
                        break

    return {"best_auroc": best_auroc}


def main(args: dict) -> None:
    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args", "wb") as file:
        pickle.dump(args, file)

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    # torch.use_deterministic_algorithms(True)

    batch_size = args["batch_size"]
    device = args["device"]
    print(f"device = {device}")

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

    # Calculate weights for each class
    class_weights = 1.0 / np.array(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_dl = get_data_loader(
        data=data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter={"correctness_value": ["C"], "phoneme_cls": list(range(1, 40))},
        class_weights=class_weights,
    )

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
    )

    n_classes = 39
    model = PhonemeClassifier(n_classes=n_classes).to(device)

    n_epochs = args["n_epochs"]
    lr = args["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
    )
    best_auroc = output["best_auroc"]


if __name__ == "__main__":
    print("in main ...")
    args = {}
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    args[
        "train_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
    args[
        "test_set_path"
    ] = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    args["n_epochs"] = 10

    for lr in [1e-2, 1e-3]:
        for batch_size in [32, 64, 128]:
            print(f"\n\nlr = {lr}")
            print(f"batch_size = {batch_size}")
            args[
                "output_dir"
            ] = f"/data/engs-pnpl/lina4471/willett2023/phoneme_classifier/PhonemeClassifier_bs_{batch_size}_lr_{lr}"
            args["lr"] = lr
            args["batch_size"] = 64

            # args["n_input_features"] = 41
            # args["n_output_features"] = 256
            # args["hidden_dim"] = 512
            # args["n_layers"] = 2
            main(args)
