import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss

    Parameters:
    real_scores : array-like
        Critic outputs for real samples
    fake_scores : array-like
        Critic outputs for fake samples

    Returns:
    float
        Wasserstein critic loss
    """

    # Convert to numpy arrays
    real_scores = np.asarray(real_scores, dtype=np.float64)
    fake_scores = np.asarray(fake_scores, dtype=np.float64)

    # Compute means
    mean_real = np.mean(real_scores)
    mean_fake = np.mean(fake_scores)

    # Compute critic loss
    loss = mean_fake - mean_real

    return float(loss)