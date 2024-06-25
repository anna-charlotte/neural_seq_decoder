from neural_decoder.neural_decoder_trainer import getDataLoader
from neural_decoder.dataset import PhonemeDataset, SpeechDataset, _padding
import pickle
import numpy as np


def test_speech_dataset():
    dataset_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/ptDecoder_ctc"
    with open(dataset_file, "rb") as handle:
        loaded_data = pickle.load(handle)

    train_dl = getDataLoader(
        data=loaded_data["train"], 
        batch_size=16, 
        collate_fn=_padding, 
        shuffle=False, 
        dataset_cls=SpeechDataset
    )

    for i, batch in enumerate(train_dl):
        X, y, X_len, y_len, dayIdx = batch
        assert X.size(0) == 16
        assert X.size(2) == 256
        assert y.size(0) == 16
        assert X_len.size(0) == 16
        assert y_len.size(0) == 16
        assert dayIdx.size(0) == 16

        if i == 2:
            break
        

def test_phoneme_speech_dataset():
    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        data = pickle.load(handle)

    dl = getDataLoader(
        data=data, batch_size=16, shuffle=False, collate_fn=None, dataset_cls=PhonemeDataset
    )

    neural_windows, phonemes, logits, dayIdx = next(iter(dl))

    assert neural_windows.size() == (16, 32, 256)
    assert phonemes.size() == (16,)
    assert logits.size() == (16, 41)
    assert dayIdx.size() == (16,)

