import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """

    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # Case 1: (N, D)
    if x.ndim == 2:
        # Mean & variance over batch axis
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Normalize
        x_hat = (x - mean) / np.sqrt(var + eps)

        # Scale and shift
        out = gamma * x_hat + beta
        return out

    # Case 2: (N, C, H, W)
    elif x.ndim == 4:
        # Mean & variance over (N, H, W) for each channel
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Normalize
        x_hat = (x - mean) / np.sqrt(var + eps)

        # Reshape gamma & beta for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        # Scale and shift
        out = gamma * x_hat + beta
        return out

    else:
        raise ValueError("Input must be (N,D) or (N,C,H,W)")