import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss

    Parameters:
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    delta : float
        Threshold parameter (> 0)

    Returns:
    float
        Mean Huber loss
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Compute error
    error = y_true - y_pred
    abs_error = np.abs(error)

    # Apply piecewise Huber formula (vectorized)
    quadratic = 0.5 * error**2
    linear = delta * (abs_error - 0.5 * delta)

    loss = np.where(abs_error <= delta, quadratic, linear)

    # Return mean loss
    return float(np.mean(loss))