import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    """
    Compute Contrastive Loss for Siamese pairs.

    Parameters:
    a : array-like, shape (N, D) or (D,)
        Embeddings from branch A
    b : array-like, shape (N, D) or (D,)
        Embeddings from branch B
    y : array-like, shape (N,)
        Labels (1 = similar, 0 = dissimilar)
    margin : float
        Margin value
    reduction : str
        "mean" or "sum"

    Returns:
    float
        Contrastive loss
    """

    # Convert to numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Handle single sample case (D,) → (1,D)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    if y.ndim == 0:
        y = y[np.newaxis]

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Compute Euclidean distance
    diff = a - b
    d = np.sqrt(np.sum(diff * diff, axis=1))

    # Compute loss
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")