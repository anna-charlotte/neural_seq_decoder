import os
import pickle
from typing import Dict

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
import numpy as np
import torch
from edit_distance import SequenceMatcher
from tqdm import tqdm

from neural_decoder.dataset import SpeechDataset, _padding, _padding_extended
from neural_decoder.neural_decoder_trainer import (
    get_data_loader,
    get_dataset_loaders,
    loadModel,
)


def cer(logits, X_len, y, y_len):
    total_edit_distance = 0
    total_seq_length = 0

    adjustedLens = X_len
    for iterIdx in range(logits.shape[0]):
        decodedSeq = torch.argmax(
            torch.tensor(logits[iterIdx, 0 : adjustedLens[iterIdx], :]),
            dim=-1,
        )  # [num_seq,]
        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
        decodedSeq = decodedSeq.cpu().detach().numpy()
        decodedSeq = np.array([i for i in decodedSeq if i != 0])

        trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
        total_edit_distance += matcher.distance()
        total_seq_length += len(trueSeq)

    return total_edit_distance / total_seq_length


def get_model_outputs(loaded_data: list[Dict], model, device):
    days = range(len(loaded_data))
    model_outputs = {
        "logits": [],
        "logitLengths": [],
        "transcriptions": [],
        "cer": [],
    }

    for dayIdx in tqdm(days):
        # Competition data days do not correspond with the index
        ds = SpeechDataset([loaded_data[dayIdx]])
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        for j, (X, y, X_len, y_len, day) in enumerate(dl):
            X, y, X_len, y_len, day = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                torch.tensor([dayIdx], dtype=torch.int64).to(device),
            )

            pred = model.forward(X, day)
            adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            for iterIdx in range(pred.shape[0]):
                model_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
                model_outputs["logitLengths"].append(adjustedLens[iterIdx].cpu().detach().item())

            # Competition data days do not correspond with the index
            transcript = loaded_data[dayIdx]["transcriptions"][j]

            model_outputs["transcriptions"].append(transcript)
            model_outputs["cer"].append(cer(pred, adjustedLens, y, y_len))

    # Logits have different length
    maxLogitLength = max([l.shape[0] for l in model_outputs["logits"]])
    model_outputs["logits"] = [
        np.pad(l, [[0, maxLogitLength - l.shape[0]], [0, 0]]) for l in model_outputs["logits"]
    ]
    model_outputs["logits"] = np.stack(model_outputs["logits"], axis=0)
    model_outputs["logitLengths"] = np.array(model_outputs["logitLengths"])
    model_outputs["transcriptions"] = np.array(model_outputs["transcriptions"])
    model_outputs["cer"] = np.array(model_outputs["cer"])

    # Shift left all phonemes!!!
    logits = model_outputs["logits"]
    model_outputs["logits"] = np.concatenate([logits[:, :, 1:], logits[:, :, :1]], axis=-1)

    return model_outputs


def save_model_output(loaded_data, model, device, out_file):
    print("In save_model_output() ... ")
    print(f"out_file = {out_file}")
    days = range(len(loaded_data))
    data = [
        {
            "sentenceDat": [],
            "phonemes": [],
            "phoneLens": [],
            "logits": [],
            "logitLengths": [],
            "transcriptions": [],
            "cer": [],
        }
        for _ in days
    ]

    for dayIdx in tqdm(days):

        # Competition data days do not correspond with the index
        ds = SpeechDataset([loaded_data[dayIdx]])
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        for j, (X, y, X_len, y_len, day) in enumerate(dl):

            X, y, X_len, y_len, day = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                torch.tensor([dayIdx], dtype=torch.int64).to(device),
            )
            data[dayIdx]["sentenceDat"].append(X[0].to("cpu"))
            data[dayIdx]["phonemes"].append(y[0].to("cpu"))
            data[dayIdx]["phoneLens"].append(y_len[0].item())

            pred = model.forward(X, day)
            adjustedLens = X_len.to("cpu")  # ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            pred = pred.to("cpu")
            for iterIdx in range(pred.shape[0]):
                data[dayIdx]["logits"].append(pred[iterIdx].cpu().detach().numpy())
                data[dayIdx]["logitLengths"].append(adjustedLens[iterIdx].cpu().detach().item())

            # Competition data days do not correspond with the index
            transcript = loaded_data[dayIdx]["transcriptions"][j]

            data[dayIdx]["transcriptions"].append(transcript)
            data[dayIdx]["cer"].append(cer(pred, adjustedLens, y, y_len))

    # Logits have different length
    maxLogitLength = max([l.shape[0] for l in data[dayIdx]["logits"]])
    data[dayIdx]["logits"] = [
        np.pad(l, [[0, maxLogitLength - l.shape[0]], [0, 0]]) for l in data[dayIdx]["logits"]
    ]
    data[dayIdx]["logits"] = np.stack(data[dayIdx]["logits"], axis=0)
    data[dayIdx]["logitLengths"] = np.array(data[dayIdx]["logitLengths"])
    data[dayIdx]["transcriptions"] = np.array(data[dayIdx]["transcriptions"])
    data[dayIdx]["cer"] = np.array(data[dayIdx]["cer"])

    # Shift left all phonemes!!!
    logits = data[dayIdx]["logits"]
    data[dayIdx]["logits"] = np.concatenate([logits[:, :, 1:], logits[:, :, :1]], axis=-1)

    with open(out_file, "wb") as handle:
        print(f"len(data) = {len(data)}")
        print(f"\nStore data to: {out_file}")
        pickle.dump(data, handle)


def inputInfo(input):
    for key in input.keys():
        if type(input[key]) == np.ndarray:
            print(key, input[key].shape, input[key].dtype, flush=True)
        else:
            print(key, type(input[key]), flush=True)


def evaluate(ngramDecoder, model_test_outputs, model_holdOut_outputs, outputFilePath):

    print("\nDecoding Test...\n", flush=True)
    decoder_out_test = lmDecoderUtils.cer_with_lm_decoder(
        ngramDecoder,
        model_test_outputs,
        outputType="speech_sil",
        blankPenalty=np.log(2),
    )

    print(f"\n-------- WER: {decoder_out_test['wer']:.3f} --------\n", flush=True)

    print("\nDecoding HoldOut...\n", flush=True)
    decoder_out_holdOut = lmDecoderUtils.cer_with_lm_decoder(
        ngramDecoder,
        model_holdOut_outputs,
        outputType="speech_sil",
        blankPenalty=np.log(2),
    )

    filename = f"{outputFilePath}_cer_{decoder_out_test['cer']:.3f}_wer_{decoder_out_test['wer']:.3f}.txt"

    print("\nSaving " + filename + " ...\n", flush=True)
    with open(filename, "w") as f:
        for decoded_transcript in decoder_out_holdOut["decoded_transcripts"]:
            f.write(decoded_transcript + "\n")


if __name__ == "__main__":
    save_output = True
    eval_model = True

    baseDir = root_directory = os.environ["DATA"] + "/willett2023"

    datsetPath = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    modelPath = "/home/lina4471/willett2023/competitionData/model/speechBaseline4"
    modelOutPath = "/home/lina4471/willett2023/competitionData/rnn"

    # Load dataset and rnn model
    with open(modelPath + "/args", "rb") as handle:
        print(f"Load args from: {modelPath}/args")
        args = pickle.load(handle)

    args["datasetPath"] = datsetPath

    trainLoaders, testLoaders, loadedData = get_dataset_loaders(args["datasetPath"], args["batchSize"])

    device = "cuda"

    model = loadModel(modelPath, device=device)
    model.to(device)
    model.eval()
    print(f"Model loaded.")

    if save_output:
        file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_train_set_with_logits.pkl"
        save_model_output(loaded_data=loadedData["train"], model=model, device=device, out_file=file)

        file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
        save_model_output(loaded_data=loadedData["test"], model=model, device=device, out_file=file)

    if eval_model:
        model_test_outputs = get_model_outputs(
            loaded_data=loadedData["test"],
            model=model,
            device=device,
        )
        print("Test raw CER: ", np.mean(model_test_outputs["cer"]), flush=True)

        model_holdOut_outputs = get_model_outputs(
            loaded_data=loadedData["competition"],
            model=model,
            device=device,
        )

        test_out_path = modelOutPath + "_test.pkl"
        holdout_out_path = modelOutPath + "_holdOut.pkl"

        with open(test_out_path, "wb") as f:
            pickle.dump(model_test_outputs, f)

        print(test_out_path + " structure:", flush=True)
        inputInfo(model_test_outputs)

        with open(holdout_out_path, "wb") as f:
            pickle.dump(model_holdOut_outputs, f)

        print(holdout_out_path + " structure:", flush=True)
        inputInfo(model_holdOut_outputs)

        # load the rnn outputs pkl for the LM
        with open(test_out_path, "rb") as handle:
            model_test_outputs = pickle.load(handle)

        print(test_out_path + " structure:", flush=True)
        inputInfo(model_test_outputs)

        with open(holdout_out_path, "rb") as handle:
            model_holdOut_outputs = pickle.load(handle)

        print(holdout_out_path + " structure:", flush=True)
        inputInfo(model_holdOut_outputs)

        # loads the language model, could take a while and requires ~60 GB of memory
        print("Load LM ...")
        lmDir = baseDir + "/languageModel"
        ngramDecoder = lmDecoderUtils.build_lm_decoder(lmDir, acoustic_scale=0.8, nbest=1, beam=18)  # 1.2
        print("LM loaded.")

        evaluate(ngramDecoder, model_test_outputs, model_holdOut_outputs, modelOutPath)
