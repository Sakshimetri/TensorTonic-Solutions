import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss using squared Euclidean distance

    Parameters:
    anchor : array-like, shape (N, D) or (D,)
    positive : array-like, shape (N, D) or (D,)
    negative : array-like, shape (N, D) or (D,)
    margin : float

    Returns:
    float
        Mean triplet loss
    """

    # Convert to numpy arrays
    anchor = np.asarray(anchor, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)

    # Handle single vector case → convert to batch of size 1
    if anchor.ndim == 1:
        anchor = anchor[np.newaxis, :]
        positive = positive[np.newaxis, :]
        negative = negative[np.newaxis, :]

    # Compute squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    # Compute triplet loss
    loss = np.maximum(0.0, d_ap - d_an + margin)

    # Return mean loss
    return float(np.mean(loss))