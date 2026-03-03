import numpy as np
from math import erf, sqrt

def gelu(x):
    # Convert input to numpy array (preserves scalar shape)
    x = np.asarray(x, dtype=float)
    
    # Vectorized GELU using erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))