import numpy as np

def tanh(x):
    # Convert input to numpy array of floats
    x = np.array(x, dtype=float)
    
    # If scalar, reshape to (1,)
    if x.ndim == 0:
        x = x.reshape(1,)
    
    # Vectorized tanh formula
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))