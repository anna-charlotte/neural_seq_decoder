# file by willet et al.
import argparse

# import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
import pickle
import re
import time

import numpy as np
import torch
from edit_distance import SequenceMatcher

from neural_decoder.dataset import SpeechDataset
from neural_decoder.neural_decoder_trainer import getDatasetLoaders, loadModel
from utils import load_args

parser = argparse.ArgumentParser(description="")
parser.add_argument("--modelPath", type=str, default=None, help="Path to model")
input_args = parser.parse_args()


# with open(input_args.modelPath + "/args", "rb") as handle:
# args = pickle.load(handle)
print(f"Load args from {input_args.modelPath + '/args'} ...")
args = load_args(input_args.modelPath + "/args")

args["datasetPath"] = "/data/engs-pnpl/lina4471/willett2023/competitionData/ptDecoder_ctc"
_, _, loadedData = getDatasetLoaders(args["datasetPath"], args["batchSize"])

print(f"\nLoad model from {input_args.modelPath} ...")
model = loadModel(input_args.modelPath, device="cpu")

device = "cpu"

model.eval()

rnn_outputs = {
    "logits": [],
    "logitLengths": [],
    "trueSeqs": [],
    "transcriptions": [],
}

partition = "competition"
print(f"\nEvaluate model on partition={partition} ...")

for i, testDayIdx in enumerate([4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]):
    # for i, testDayIdx in enumerate(range(len(loadedData[partition]))):
    print(f"testDayIdx = {testDayIdx}")
    print(f"len([loadedData[partition]]) = {len([loadedData[partition]])}")
    print(f"type([loadedData[partition]]) = {type([loadedData[partition]])}")
    # test_ds = SpeechDataset([loadedData[partition][testDayIdx]])
    test_ds = SpeechDataset([loadedData[partition][i]])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            torch.tensor([testDayIdx], dtype=torch.int64).to(device),
        )
        pred = model.forward(X, dayIdx)
        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

        if j == 0:
            print(f"type(y) = {type(y)}")

        for iterIdx in range(pred.shape[0]):
            trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

            logits = pred[iterIdx].cpu().detach().numpy()
            rnn_outputs["logits"].append(logits)
            rnn_outputs["logitLengths"].append(adjustedLens[iterIdx].cpu().detach().item())
            rnn_outputs["trueSeqs"].append(trueSeq)
            if j == 0 and iterIdx == 0:
                print(f"trueSeq = {trueSeq}")
                print(f"trueSeq.shape = {trueSeq.shape}")
                print(f"logits.shape = {logits.shape}")

        transcript = loadedData[partition][i]["transcriptions"][j].strip()
        transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
        transcript = transcript.replace("--", "").lower()
        rnn_outputs["transcriptions"].append(transcript)


# here just compute the cer

# MODEL_CACHE_DIR = "/scratch/users/stfan/huggingface"
# # Load OPT 6B model
# llm, llm_tokenizer = lmDecoderUtils.build_opt(
#     cacheDir=MODEL_CACHE_DIR, device="auto", load_in_8bit=True
# )

# lmDir = "/data/engs-pnpl/lina4471/willett2023/languageModel"
# ngramDecoder = lmDecoderUtils.build_lm_decoder(
#     lmDir, acoustic_scale=0.5, nbest=100, beam=18
# )


# # LM decoding hyperparameters
# acoustic_scale = 0.5
# blank_penalty = np.log(7)
# llm_weight = 0.5

# llm_outputs = []
# # Generate nbest outputs from 5gram LM
# start_t = time.time()
# nbest_outputs = []
# for j in range(len(rnn_outputs["logits"])):
#     logits = rnn_outputs["logits"][j]
#     logits = np.concatenate(
#         [logits[:, 1:], logits[:, 0:1]], axis=-1
#     )  # Blank is last token
#     logits = lmDecoderUtils.rearrange_speech_logits(logits[None, :, :], has_sil=True)
#     nbest = lmDecoderUtils.lm_decode(
#         ngramDecoder,
#         logits[0],
#         blankPenalty=blank_penalty,
#         returnNBest=True,
#         rescore=True,
#     )
#     nbest_outputs.append(nbest)
# time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
# print(f"5gram decoding took {time_per_sample} seconds per sample")

# for i in range(len(rnn_outputs["transcriptions"])):
#     new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
#     rnn_outputs["transcriptions"][i] = np.array(new_trans)

# # Rescore nbest outputs with LLM
# start_t = time.time()
# llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
#     llm,
#     llm_tokenizer,
#     nbest_outputs[:],
#     acoustic_scale,
#     rnn_outputs,
#     outputType="speech_sil",
#     returnCI=True,
#     lengthPenalty=0,
#     alpha=llm_weight,
# )
# # time_per_sample = (time.time() - start_t) / len(logits)
# print(f"LLM decoding took {time_per_sample} seconds per sample")

# print(llm_out["cer"], llm_out["wer"])
# with open(input_args.modelPath + "/llm_out", "wb") as handle:
#     pickle.dump(llm_out, handle)

# decodedTranscriptions = llm_out["decoded_transcripts"]
# with open(input_args.modelPath + "/5gramLLMCompetitionSubmission.txt", "w") as f:
#     for x in range(len(decodedTranscriptions)):
#         f.write(decodedTranscriptions[x] + "\n")
