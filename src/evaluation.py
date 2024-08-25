import numpy as np
import torch
from scipy.signal import correlate2d
from scipy.stats import multivariate_normal
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from typing import Tuple

from scipy.stats import mannwhitneyu


from text2brain.models.vae_interface import VAEBase


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix for the given data.

    Parameters:
    data (numpy.ndarray): A 2D array of shape (n_channels, n_timesteps).

    Returns:
    numpy.ndarray: A 2D array representing the correlation matrix.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of numpy.ndarray.")
    if data.shape != (256, 32):
        raise ValueError(f"Input data must be of shape (256, 256), but it is: {data.shape}")
    if np.isnan(data).any():
        raise ValueError(
            "Input data contains NaNs. Please handle them before computing the correlation matrix."
        )

    variances = np.var(data, axis=1)
    if np.any(variances == 0):
        data = np.where(variances[:, None] == 0, data + np.random.normal(0, 1e-10, data.shape), data)

    corr_matrix = np.corrcoef(data)
    return corr_matrix


def compute_cross_correlation(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    assert arr1.dim() == 2, f"Array is of shape {arr1.shape}, should be 2-dim."
    assert arr1.size() == arr2.size(), f"Given arrays are not of same shape: {arr1.shape} != {arr2.shape}"
    cross_corr = correlate2d(arr1, arr2, mode="full")
    return cross_corr


def compute_likelihood(mean: torch.Tensor, logvar: torch.Tensor, x: torch.Tensor, vae: VAEBase):
    print(f"mean.size() = {mean.size()}")
    print(f"logvar.size() = {logvar.size()}")
    print(f"x.size() = {x.size()}")

    x_mean, x_logvar = vae.encode(x)

    var = torch.exp(logvar)
    std = torch.sqrt(var)
    cov = torch.diag_embed(std)
    mvn = multivariate_normal(mean.numpy(), cov.numpy())

    return mvn.logpdf(x.numpy())


def compute_auroc_with_stderr(y_true: np.ndarray, y_pred: np.ndarray, n_iters: int = 1_000) -> Tuple[np.ndarray, np.ndarray]:
    auroc_scores = []
    for _ in range(n_iters):
        y_true_bootstrap, y_pred_bootstrap = resample(y_true, y_pred)
    
        auroc = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
        auroc_scores.append(auroc)

    mean_auroc = np.mean(auroc_scores)
    std_auroc = np.std(auroc_scores, ddof=1)
    sem_auroc = std_auroc / np.sqrt(len(auroc_scores))

    return mean_auroc, sem_auroc
    # scipy.stats.mannwhitneyu(x,y, alternative='two-sided')
    # scipy.stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')


def compute_man_whitney(y_true: np.ndarray, y_pred_1: np.ndarray, y_pred_2: np.ndarray, n_iters: int = 1_000) -> Tuple[np.ndarray, np.ndarray]:
    auroc_scores_1 = []
    auroc_scores_2 = []

    for _ in range(n_iters):
        y_true_bootstrap, y_pred_1_bootstrap = resample(y_true, y_pred_1)
        auroc_1 = roc_auc_score(y_true_bootstrap, y_pred_1_bootstrap)
        auroc_scores_1.append(auroc_1)

        y_true_bootstrap, y_pred_2_bootstrap = resample(y_true, y_pred_2)
        auroc_2 = roc_auc_score(y_true_bootstrap, y_pred_2_bootstrap)
        auroc_scores_2.append(auroc_2)

    stat, p_value = mannwhitneyu(auroc_scores_1, auroc_scores_2, alternative='two-sided')

    return stat, p_value
    # scipy.stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')