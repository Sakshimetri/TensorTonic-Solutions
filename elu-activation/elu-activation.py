import numpy as np

def elu(x, alpha):
    """
    Applies the ELU activation function.

    Parameters:
        x : list (input values)
        alpha : float (alpha >= 0)

    Returns:
        list of floats after applying ELU
    """
    x = np.asarray(x, dtype=float)
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return result.tolist()