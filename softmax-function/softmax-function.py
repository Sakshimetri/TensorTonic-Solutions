import numpy as np

def softmax(x):
    """
    Compute the Softmax of input array (1D or 2D).

    Parameters:
        x : np.ndarray

    Returns:
        np.ndarray (same shape as input)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    elif x.ndim == 2:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    else:
        raise ValueError("Input must be 1D or 2D NumPy array")