import numpy as np


def compute_correlation_matrix(data: np.ndarray):
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
