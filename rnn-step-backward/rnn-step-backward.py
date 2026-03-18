import numpy as np

def rnn_step_backward(dh, cache):
    """
    Backward pass for one RNN step (tanh).

    Returns:
        dx_t: (D,)
        dh_prev: (H,)
        dW: (H, D)
        dU: (H, H)
        db: (H,)
    """

    # Unpack cache
    x_t, h_prev, h_t, W, U, b = cache

    # Convert to numpy
    dh = np.asarray(dh, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    h_t = np.asarray(h_t, dtype=float)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)

    # ---- Step 1: tanh derivative ----
    dz = dh * (1 - h_t**2)   # (H,)

    # ---- Step 2: gradients ----
    dx_t = dz @ W            # (D,)
    dh_prev = dz @ U         # (H,)

    dW = np.outer(dz, x_t)   # (H, D)
    dU = np.outer(dz, h_prev)# (H, H)

    db = dz                  # (H,)

    return dx_t, dh_prev, dW, dU, db
