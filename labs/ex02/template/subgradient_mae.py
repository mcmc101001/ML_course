import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    N = np.shape(y)[0]
    grad = np.zeros(2)
    e = y - tx @ w
    grad[0] = -np.sum(np.sign(e)) / N
    grad[1] = -np.sign(e) @ tx[:, 1] / N

    return grad
