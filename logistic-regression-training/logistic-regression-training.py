import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    # Convert inputs to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    # Training loop
    for _ in range(steps):
        
        # Linear combination
        z = X @ w + b
        
        # Sigmoid prediction
        p = _sigmoid(z)
        
        # Gradients
        dw = (X.T @ (p - y)) / N
        db = np.sum(p - y) / N
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    # Return exactly this format
    return (w, float(b))