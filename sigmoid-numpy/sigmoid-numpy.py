import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
        x : scalar, list, or numpy array

    Returns:
        numpy array of floats
    """
    # Convert input to NumPy array of floats
    x = np.asarray(x, dtype=np.float64)
    
    # Compute sigmoid (vectorized)
    return 1 / (1 + np.exp(-x))