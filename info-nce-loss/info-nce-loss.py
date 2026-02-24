import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss

    Parameters:
    Z1 : array-like, shape (N, D)
        First embedding batch
    Z2 : array-like, shape (N, D)
        Second embedding batch
    temperature : float
        Temperature parameter (tau > 0)

    Returns:
    float
        Mean InfoNCE loss
    """

    # Convert to numpy arrays
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)

    # Compute similarity matrix (N x N)
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability: subtract row max
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    # Compute exp(similarity)
    exp_S = np.exp(S_stable)

    # Compute softmax denominator
    denom = np.sum(exp_S, axis=1)

    # Positive similarities are diagonal elements
    pos = np.diag(exp_S)

    # Compute loss
    loss = -np.log(pos / denom)

    # Return mean loss
    return float(np.mean(loss))