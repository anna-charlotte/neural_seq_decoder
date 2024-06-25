import os
import pickle
import time
from typing import Any, Tuple, Type

import hydra
import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.utils.data import DataLoader

from .dataset import ExtendedSpeechDataset, SpeechDataset, _padding, _padding_extended
from .model import GRUDecoder


def getDataLoader(
    data: dict,
    batch_size: int,
    shuffle: bool,
    collate_fn: callable,
    transform: callable = None,
    dataset_cls: Type[Any] = SpeechDataset,
) -> DataLoader:

    ds = dataset_cls(data, transform=transform)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    return dl


def getDatasetLoaders(
    datasetName: str, batchSize: int, dataset_cls: Type[Any] = SpeechDataset,
) -> Tuple[DataLoader, DataLoader, dict]:
    print("In getDatasetLoaders()")
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    padding_fnc = None
    if dataset_cls == SpeechDataset:
        padding_fnc = _padding
    elif dataset_cls == ExtendedSpeechDataset:
        padding_fnc = _padding_extended

    train_dl = getDataLoader(
        data=loadedData["train"],
        batch_size=batchSize,
        shuffle=True,
        collate_fn=padding_fnc,
        dataset_cls=dataset_cls,
    )
    test_dl = getDataLoader(
        data=loadedData["test"],
        batch_size=batchSize,
        shuffle=True,
        collate_fn=padding_fnc,
        dataset_cls=dataset_cls,
    )

    return train_dl, test_dl, loadedData


def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(args["datasetPath"], args["batchSize"],)
    if "datasetPathSynthetic" in args.keys() and args["datasetPathSynthetic"] != "":
        with open(datasetName, "rb") as handle:
            data = pickle.load(handle)

        syntheticLoader = getDataLoader(
            data=data, batch_size=args["batchSize"], shuffle=True, collate_fn=_padding,
        )
        assert (
            0.0 <= args["proportionSynthetic"] <= 1.0
        ), "The value for the proportion of synthetic data is not in the range 0.0 to 1.0."
        propSynthetic = 1 - args["proportionSynthetic"]

        trainLoader = MergedDataLoader(loader1=trainLoader, loader2=syntheticLoader, prop1=propSynthetic)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lrStart"], betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args["lrEnd"] / args["lrStart"], total_iters=args["nBatch"],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
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

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
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
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]), dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

                        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


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
