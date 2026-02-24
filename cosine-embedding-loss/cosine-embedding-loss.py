import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    """
    Compute Cosine Embedding Loss

    Parameters:
    x1 : array-like
        First vector
    x2 : array-like
        Second vector
    label : int
        1 for similar, -1 for dissimilar
    margin : float
        Margin for dissimilar pairs (>= 0)

    Returns:
    float
        Cosine embedding loss
    """

    # Convert to numpy arrays
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    # Compute cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    cos_sim = dot / (norm1 * norm2)

    # Compute loss
    if label == 1:
        loss = 1.0 - cos_sim
    elif label == -1:
        loss = max(0.0, cos_sim - margin)
    else:
        raise ValueError("label must be 1 or -1")

    return float(loss)