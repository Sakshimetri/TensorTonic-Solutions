import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x, dtype=float)

    # Case 1: (C, H, W)
    if x.ndim == 3:
        return np.mean(x, axis=(1, 2))

    # Case 2: (N, C, H, W)
    elif x.ndim == 4:
        return np.mean(x, axis=(2, 3))

    else:
        raise ValueError("Input must be of shape (C,H,W) or (N,C,H,W)")