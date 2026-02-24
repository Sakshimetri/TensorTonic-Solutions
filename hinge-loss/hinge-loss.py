import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    """
    Compute Binary Hinge Loss for SVM

    Parameters:
    y_true : array-like, shape (N,)
        True labels {-1, +1}
    y_score : array-like, shape (N,)
        Predicted scores (real-valued)
    margin : float
        Margin parameter (default=1.0)
    reduction : str
        "mean" or "sum"

    Returns:
    float
        Hinge loss
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have same shape")

    # Validate labels
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must contain only -1 and +1")

    # Compute hinge loss vectorized
    loss = np.maximum(0.0, margin - y_true * y_score)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")