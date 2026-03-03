import numpy as np

def leaky_relu(x, alpha=0.1):
    """
    Applies the Leaky ReLU activation function.

    Parameters:
        x : scalar, list, or numpy array
        alpha : float (slope for negative values)

    Returns:
        numpy array after applying Leaky ReLU
    """
    x = np.asarray(x, dtype=float)   # Convert input to NumPy array
    return np.where(x >= 0, x, alpha * x)