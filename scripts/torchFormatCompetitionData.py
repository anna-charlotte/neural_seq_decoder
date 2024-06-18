import os
import re
from g2p_en import G2p
import numpy as np
import scipy
import scipy.io
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def phoneToId(p):
    return PHONE_DEF_SIL.index(p)


def loadFeaturesAndNormalize(sessionPath):

    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    frame_lens = []
    block_means = []
    block_stds = []
    n_trials = dat["sentenceText"].shape[0]

    # collect area 6v tx1 and spikePow features
    for i in range(n_trials):
        # get time series of TX and spike power for this trial
        # first 128 columns = area 6v only
        features = np.concatenate(
            [dat["tx1"][0, i][:, 0:128], dat["spikePow"][0, i][:, 0:128]], axis=1
        )

        sentence_len = features.shape[0]
        sentence = dat["sentenceText"][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    # block-wise feature normalization
    blockNums = np.squeeze(dat["blockIdx"])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums == blockList[b])
        sentIdx = sentIdx[:, 0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate(
            input_features[blocks[b][0] : (blocks[b][-1] + 1)], axis=0
        )
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    # convert to tfRecord file
    session_data = {
        "inputFeatures": input_features,
        "transcriptions": transcriptions,
        "frameLens": frame_lens,
    }

    return session_data


def _convert_to_ascii(text):
    return [ord(char) for char in text]


def getDataset(fileName):
    session_data = loadFeaturesAndNormalize(fileName)

    allDat = []
    trueSentences = []
    seqElements = []

    for x in range(len(session_data["inputFeatures"])):
        allDat.append(session_data["inputFeatures"][x])

        thisTranscription = str(session_data["transcriptions"][x]).strip()
        thisTranscription = re.sub(r"[^a-zA-Z\- \']", "", thisTranscription)
        thisTranscription = thisTranscription.replace("--", "").lower()
        addInterWordSymbol = True

        phonemes = []
        for p in g2p(thisTranscription):
            if addInterWordSymbol and p == " ":
                phonemes.append("SIL")
            p = re.sub(r"[0-9]", "", p)  # Remove stress
            if re.match(r"[A-Z]+", p):  # Only keep phonemes
                phonemes.append(p)

        # add one SIL symbol at the end so there's one at the end of each word
        if addInterWordSymbol:
            phonemes.append("SIL")

        seqLen = len(phonemes)
        maxSeqLen = 500
        seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)
        seqClassIDs[0:seqLen] = [
            phoneToId(p) + 1 for p in phonemes
        ]  # + 1 to be consistent with the pipeline, the shifted on eval
        seqElements.append(seqClassIDs)
        paddedTranscription = np.zeros([maxSeqLen]).astype(np.int32)
        paddedTranscription[0 : len(thisTranscription)] = np.array(
            _convert_to_ascii(thisTranscription)
        )
        trueSentences.append(paddedTranscription)
        t = session_data["transcriptions"][x]
        print(f"allDat = {allDat}")

    newDataset = {}
    newDataset["sentenceDat"] = allDat
    newDataset["transcriptions"] = trueSentences
    newDataset["phonemes"] = seqElements

    timeSeriesLens = []
    phoneLens = []
    for x in range(len(newDataset["sentenceDat"])):
        timeSeriesLens.append(newDataset["sentenceDat"][x].shape[0])

        zeroIdx = np.argwhere(newDataset["phonemes"][x] == 0)
        phoneLens.append(zeroIdx[0, 0])

    newDataset["timeSeriesLens"] = np.array(timeSeriesLens)
    newDataset["phoneLens"] = np.array(phoneLens)
    newDataset["phonePerTime"] = newDataset["phoneLens"].astype(
        np.float32
    ) / newDataset["timeSeriesLens"].astype(np.float32)
    return newDataset


if __name__ == "__main__":

    root_directory = os.environ["DATA"]
    os.sys.path.append(root_directory)
    print(f"root_directory = {root_directory}")

    datasets = "/willett2023/competitionData"
    sessionNames = os.listdir(root_directory + datasets + "/train/")
    sessionNames = [name.replace(".mat", "") for name in sessionNames]

    sessionNames.sort()
    print(f"sessionNames = {sessionNames}")

    g2p = G2p()
    PHONE_DEF = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    ]
    PHONE_DEF_SIL = PHONE_DEF + ["SIL"]

    trainDatasets = []
    testDatasets = []
    competitionDatasets = []

    dataDir = root_directory + datasets
    print(f"dataDir = {dataDir}")

    for dayIdx in tqdm(range(len(sessionNames))):
        trainDataset = getDataset(dataDir + "/train/" + sessionNames[dayIdx] + ".mat")
        testDataset = getDataset(dataDir + "/test/" + sessionNames[dayIdx] + ".mat")

        trainDatasets.append(trainDataset)
        testDatasets.append(testDataset)

        if os.path.exists(
            dataDir + "/competitionHoldOut/" + sessionNames[dayIdx] + ".mat"
        ):
            dataset = getDataset(
                dataDir + "/competitionHoldOut/" + sessionNames[dayIdx] + ".mat"
            )
            competitionDatasets.append(dataset)

    competitionDays = []
    for dayIdx in range(len(sessionNames)):
        if os.path.exists(
            dataDir + "/competitionHoldOut/" + sessionNames[dayIdx] + ".mat"
        ):
            competitionDays.append(dayIdx)
    print(f"competitionDays = {competitionDays}")

    allDatasets = {}
    allDatasets["train"] = trainDatasets
    allDatasets["test"] = testDatasets
    allDatasets["competition"] = competitionDatasets

    # file = root_directory + datasets + "/pytorchTFRecords.pkl"
    # with open(file, "wb") as handle:
    #     print(f"Write files to: {file}")
    #     pickle.dump(allDatasets, handle)
