import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Computes average cross-entropy loss for multi-class classification.

    Parameters:
    y_true : array-like of shape (N,)
        Correct class indices
    y_pred : array-like of shape (N, K)
        Predicted probabilities

    Returns:
    float
        Average cross-entropy loss
    """
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Number of samples
    N = y_true.shape[0]
    
    # Select probability of correct class for each sample
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss