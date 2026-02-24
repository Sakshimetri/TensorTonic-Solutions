import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE)

    Parameters:
    y_pred : array-like, shape (N,)
        Predicted values
    y_true : array-like, shape (N,)
        True target values

    Returns:
    float
        Mean Squared Error
    """

    # Convert to numpy arrays
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Compute MSE
    mse = np.mean((y_pred - y_true) ** 2)

    return mse