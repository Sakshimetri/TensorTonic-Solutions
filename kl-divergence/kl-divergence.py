import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q)

    Parameters:
    p : array-like, shape (N,)
        Reference probability distribution
    q : array-like, shape (N,)
        Approximation probability distribution
    eps : float
        Small value to prevent log(0)

    Returns:
    float
        KL divergence
    """

    # Convert to numpy arrays
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Add epsilon to q for numerical stability
    q = q + eps

    # Only compute where p > 0 (since p*log(p/q)=0 when p=0)
    mask = p > 0

    # Compute KL divergence
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return float(kl)