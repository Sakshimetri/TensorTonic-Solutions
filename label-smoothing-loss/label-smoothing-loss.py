import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1):
    """
    Compute Label Smoothing Cross-Entropy Loss

    Parameters:
    predictions : array-like, shape (K,)
        Predicted probabilities for each class
    target : int
        Correct class index
    epsilon : float
        Smoothing parameter (0 <= epsilon <= 1)

    Returns:
    float
        Label smoothing loss
    """

    # Convert to numpy array
    p = np.asarray(predictions, dtype=np.float64)

    # Number of classes
    K = p.shape[0]

    # Build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # Compute cross-entropy loss
    loss = -np.sum(q * np.log(p))

    return float(loss)