import numpy as np

def swish(x):
    """
    Applies the Swish activation function:
        Swish(x) = x * sigmoid(x)

    Returns:
        np.ndarray of floats (shape preserved)
    """

    x = np.asarray(x, dtype=float)

    # Numerically stable sigmoid
    sigmoid = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

    return x * sigmoid