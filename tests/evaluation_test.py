import pickle
from datetime import datetime
import numpy as np

from data.dataset import PhonemeDataset, SpeechDataset, _padding
from evaluation import compute_correlation_matrix
from neural_decoder.neural_decoder_trainer import get_data_loader
from neural_decoder.transforms import TransposeTransform
from neural_decoder.phoneme_utils import ROOT_DIR
from text2brain.visualization import plot_correlation_matrix



def test_compute_correlation_matrix():
    test_file = "/data/engs-pnpl/lina4471/willett2023/competitionData/rnn_test_set_with_logits.pkl"
    with open(test_file, "rb") as handle:
        data = pickle.load(handle)

    transform = TransposeTransform(dim1=0, dim2=1)
    dl = get_data_loader(data=data, batch_size=128, shuffle=True, collate_fn=None, transform=transform, dataset_cls=PhonemeDataset)

    neural_windows, phonemes, _, _ = next(iter(dl))
    for X, y in zip(neural_windows, phonemes):

        corr_matrix = compute_correlation_matrix(X.cpu().detach().numpy())

        assert corr_matrix.shape == (256, 256)
        assert not np.isnan(corr_matrix).any(), "The correlation matrix contains NaNs."
        assert np.all((corr_matrix >= -1) & (corr_matrix <= 1)), "The correlation matrix contains values outside the range [-1, 1]."
    
        # now = datetime.now()
        # timestamp = now.strftime("%Y%m%d_%H%M%S")
        # plot_correlation_matrix(
        #     corr_matrix=corr_matrix,
        #     out_file=ROOT_DIR / "evaluation" / "correlation_plots" / f"corr__matrix_phoneme_cls_{y}__{timestamp}.png",
        #     xlabel="Channels",
        #     ylabel="Channels",
        #     title=f"Corrleation matrix, phoneme class {y}"
        # )

# test_compute_correlation_matrix()