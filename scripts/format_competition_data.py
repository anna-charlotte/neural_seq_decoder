# file by willet et al.
import os
import pickle
import re
from typing import List

import numpy as np
import scipy
import scipy.io
from g2p_en import G2p
from tqdm import tqdm

from neural_decoder.phoneme_utils import phone_to_id

g2p = G2p()


def load_features_and_normalize(session_path: str) -> dict:
    dat = scipy.io.loadmat(session_path)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = dat["sentenceText"].shape[0]

    # collect area 6v tx1 and spikePow features
    for i in range(n_trials):
        # get time series of TX and spike power for this trial
        # first 128 columns = area 6v only
        features = np.concatenate([dat["tx1"][0, i][:, 0:128], dat["spikePow"][0, i][:, 0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat["sentenceText"][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    # block-wise feature normalization
    block_nums = np.squeeze(dat["blockIdx"])
    block_list = np.unique(block_nums)
    blocks = []
    for b in range(len(block_list)):
        sentIdx = np.argwhere(block_nums == block_list[b])
        sentIdx = sentIdx[:, 0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0] : (blocks[b][-1] + 1)], axis=0)
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


def _convert_to_ascii(text: str) -> List[int]:
    return [ord(char) for char in text]


def get_dataset(file_name):
    session_data = load_features_and_normalize(file_name)

    data = []
    true_sentences = []
    seq_elements = []

    for x in range(len(session_data["inputFeatures"])):
        data.append(session_data["inputFeatures"][x])

        transcription = str(session_data["transcriptions"][x]).strip()
        transcription = re.sub(r"[^a-zA-Z\- \']", "", transcription)
        transcription = transcription.replace("--", "").lower()
        add_inter_word_symbol = True

        phonemes = []
        for p in g2p(transcription):
            if add_inter_word_symbol and p == " ":
                phonemes.append("SIL")
            p = re.sub(r"[0-9]", "", p)  # Remove stress
            if re.match(r"[A-Z]+", p):  # Only keep phonemes
                phonemes.append(p)

        # add one SIL symbol at the end so there's one at the end of each word
        if add_inter_word_symbol:
            phonemes.append("SIL")

        seq_len = len(phonemes)
        max_seq_len = 500
        seq_class_ids = np.zeros([max_seq_len]).astype(np.int32)
        seq_class_ids[0:seq_len] = [
            phone_to_id(p) + 1 for p in phonemes
        ]  # + 1 to be consistent with the pipeline, the shifted on eval
        seq_elements.append(seq_class_ids)
        padded_transcription = np.zeros([max_seq_len]).astype(np.int32)
        padded_transcription[0 : len(transcription)] = np.array(_convert_to_ascii(transcription))
        true_sentences.append(padded_transcription)

    out_data = {}
    out_data["sentenceDat"] = data
    out_data["transcriptions"] = true_sentences
    out_data["phonemes"] = seq_elements

    time_series_len = []
    phone_len = []
    for x in range(len(out_data["sentenceDat"])):
        time_series_len.append(out_data["sentenceDat"][x].shape[0])

        zero_idx = np.argwhere(out_data["phonemes"][x] == 0)
        phone_len.append(zero_idx[0, 0])

    out_data["timeSeriesLens"] = np.array(time_series_len)
    out_data["phoneLens"] = np.array(phone_len)
    out_data["phonePerTime"] = out_data["phoneLens"].astype(np.float32) / out_data["timeSeriesLens"].astype(
        np.float32
    )
    return out_data


if __name__ == "__main__":
    root_directory = os.environ["DATA"]
    os.sys.path.append(root_directory)
    print(f"root_directory = {root_directory}")

    datasets = "/willett2023/competitionData"
    session_names = os.listdir(root_directory + datasets + "/train/")
    session_names = [name.replace(".mat", "") for name in session_names]

    session_names.sort()
    print(f"session_names = {session_names}")

    train_ds = []
    test_ds = []
    competition_ds = []

    data_dir = root_directory + datasets
    print(f"data_dir = {data_dir}")

    for dayIdx in tqdm(range(len(session_names))):
        train_set = get_dataset(data_dir + "/train/" + session_names[dayIdx] + ".mat")
        test_set = get_dataset(data_dir + "/test/" + session_names[dayIdx] + ".mat")

        train_ds.append(train_set)
        test_ds.append(test_set)

        if os.path.exists(data_dir + "/competitionHoldOut/" + session_names[dayIdx] + ".mat"):
            dataset = get_dataset(data_dir + "/competitionHoldOut/" + session_names[dayIdx] + ".mat")
            competition_ds.append(dataset)

    competition_days = []
    for dayIdx in range(len(session_names)):
        if os.path.exists(data_dir + "/competitionHoldOut/" + session_names[dayIdx] + ".mat"):
            competition_days.append(dayIdx)
    print(f"competition_days = {competition_days}")

    all_datasets = {}
    all_datasets["train"] = train_ds
    all_datasets["test"] = test_ds
    all_datasets["competition"] = competition_ds

    file = root_directory + datasets + "/pytorchTFRecords.pkl"
    with open(file, "wb") as handle:
        print(f"Write files to: {file}")
        pickle.dump(all_datasets, handle)
