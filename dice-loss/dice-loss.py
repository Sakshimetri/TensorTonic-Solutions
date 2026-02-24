import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation

    Parameters:
    p : array-like
        Predicted probabilities, shape (N,) or (H,W) or any shape
    y : array-like
        Ground truth binary mask, same shape as p
    eps : float
        Smoothing epsilon

    Returns:
    float
        Dice loss
    """

    # Convert to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Flatten arrays (works for any shape)
    p = p.ravel()
    y = y.ravel()

    # Compute Dice coefficient
    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    total_y = np.sum(y)

    dice = (2 * intersection + eps) / (sum_p + total_y + eps)

    # Dice loss
    return 1 - dice