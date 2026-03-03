import numpy as np

def relu(x):
    # Convert to numpy array of floats
    x = np.asarray(x, dtype=float)
    
    # Vectorized ReLU
    return np.maximum(0, x)