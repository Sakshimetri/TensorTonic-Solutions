import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    Compute Binary Focal Loss

    Parameters:
    predictions : array-like, shape (N,)
        Predicted probabilities (0 < p < 1)
    targets : array-like, shape (N,)
        Binary targets (0 or 1)
    alpha : float
        Balancing factor (> 0)
    gamma : float
        Focusing parameter (>= 0)

    Returns:
    float
        Mean binary focal loss
    """

    # Convert to numpy arrays
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Compute p_t
    p_t = np.where(targets == 1, predictions, 1 - predictions)

    # Compute focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    # Return mean loss
    return float(np.mean(loss))