import json
import os
import pickle
import time
from typing import Any, Tuple, Type

import hydra
import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.utils.data import DataLoader, WeightedRandomSampler

from neural_decoder.data.dataloader import MergedDataLoader
from neural_decoder.data.dataset import PhonemeDataset, SpeechDataset, _padding
from neural_decoder.model import GRUDecoder


def get_data_loader(
    data: dict,
    batch_size: int,
    shuffle: bool,
    collate_fn: callable,
    transform: callable = None,
    dataset_cls: Type[Any] = SpeechDataset,
    phoneme_ds_filter: dict = None,
    class_weights=None,
) -> DataLoader:
    if phoneme_ds_filter is None:
        phoneme_ds_filter = {}

    if dataset_cls == SpeechDataset:
        ds = SpeechDataset(data, transform=transform)
    elif dataset_cls == PhonemeDataset:
        ds = PhonemeDataset(data, transform=transform, filter_by=phoneme_ds_filter)
    else:
        raise ValueError(f"Given dataset_cls is not valid: {dataset_cls.__name__}")

    sampler = None
    if class_weights is not None:
        all_y = ds.phonemes
        sample_weights = class_weights[torch.tensor(all_y) - 1]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    if class_weights is not None and shuffle:
        raise ValueError("class_weights option is mutually exclusive with shuffle option.")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dl


def get_dataset_loaders(
    dataset_name: str, batch_size: int, dataset_cls: Type[Any] = SpeechDataset, phoneme_ds_filter: dict = {}
) -> Tuple[DataLoader, DataLoader, dict]:
    print("In get_dataset_loaders()")
    with open(dataset_name, "rb") as handle:
        loaded_data = pickle.load(handle)

    padding_fnc = None
    if dataset_cls == SpeechDataset:
        padding_fnc = _padding
    elif dataset_cls == PhonemeDataset:
        padding_fnc = None
    else:
        raise ValueError(f"Given dataset class is not valid: {dataset_cls}")

    train_dl = get_data_loader(
        data=loaded_data["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=padding_fnc,
        dataset_cls=dataset_cls,
        phoneme_ds_filter=phoneme_ds_filter,
    )
    test_dl = get_data_loader(
        data=loaded_data["test"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=padding_fnc,
        dataset_cls=dataset_cls,
        phoneme_ds_filter=phoneme_ds_filter,
    )

    return train_dl, test_dl, loaded_data


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    with open(args["outputDir"] / "args.json", "w") as file:
        json.dump(args, file, indent=4)

    train_loader, test_loader, loaded_data = get_dataset_loaders(
        args["datasetPath"],
        args["batchSize"],
    )
    if "datasetPathSynthetic" in args.keys() and args["datasetPathSynthetic"] != "":
        dataset_name = args["datasetPathSynthetic"]
        with open(dataset_name, "rb") as handle:
            data = pickle.load(handle)

        synthetic_loader = get_data_loader(
            data=data,
            batch_size=args["batchSize"],
            shuffle=True,
            collate_fn=_padding,
        )
        assert (
            0.0 <= args["proportionSynthetic"] <= 1.0
        ), "The value for the proportion of synthetic data is not in the range 0.0 to 1.0."
        prop_synthetic = 1 - args["proportionSynthetic"]

        train_loader = MergedDataLoader(loader1=train_loader, loader2=synthetic_loader, prop1=prop_synthetic)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loaded_data["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    test_loss = []
    test_cer = []
    start_time = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(train_loader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"]

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(end_time - start_time)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                all_loss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in test_loader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    all_loss.append(loss.cpu().detach().numpy())

                    adjusted_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                    for iterIdx in range(pred.shape[0]):
                        decoded_seq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjusted_lens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                        decoded_seq = decoded_seq.cpu().detach().numpy()
                        decoded_seq = np.array([i for i in decoded_seq if i != 0])

                        true_seq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

                        matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded_seq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(true_seq)

                avg_day_loss = np.sum(all_loss) / len(test_loader)
                cer = total_edit_distance / total_seq_length

                end_time = time.time()
                print(
                    f"batch {batch}, ctc loss: {avg_day_loss:>7f}, cer: {cer:>7f}, time/batch: {(end_time - start_time)/100:>7.3f}"
                )
                start_time = time.time()

            if len(test_cer) > 0 and cer < np.min(test_cer):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            test_loss.append(avg_day_loss)
            test_cer.append(cer)

            t_stats = {}
            t_stats["test_loss"] = np.array(test_loss)
            t_stats["test_cer"] = np.array(test_cer)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(t_stats, file)
            with open(args["outputDir"] / "trainingStats.json", "w") as file:
                json.dump(args, file, indent=4)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)


if __name__ == "__main__":
    main()
