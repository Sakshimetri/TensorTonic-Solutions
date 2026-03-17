import numpy as np

def clip_gradients(g, max_norm):
    g = np.asarray(g, dtype=float)

    # Handle edge case: non-positive max_norm → no clipping
    if max_norm <= 0:
        return g

    norm = np.linalg.norm(g)

    # No clipping needed
    if norm == 0 or norm <= max_norm:
        return g

    # Apply scaling
    scale = max_norm / norm
    return g * scale