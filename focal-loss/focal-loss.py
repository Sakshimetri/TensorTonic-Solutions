import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Binary Focal Loss

    Parameters:
    p : np.ndarray, shape (N,)
        Predicted probabilities (0 < p < 1)
    y : np.ndarray, shape (N,)
        True binary labels (0 or 1)
    gamma : float
        Focusing parameter (>= 0)

    Returns:
    float
        Mean focal loss
    """

    # Convert to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Compute focal loss (vectorized)
    loss = -( (1 - p)**gamma * y * np.log(p) +
              p**gamma * (1 - y) * np.log(1 - p) )

    # Return mean loss
    return np.mean(loss)