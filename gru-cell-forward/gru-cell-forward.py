import numpy as np

def gru_cell_forward(x, h_prev, params):
    """
    GRU forward pass for one time step.
    Supports (D,) & (H,) or (N,D) & (N,H).
    """

    # --- Helper functions (define inside to avoid errors) ---
    def sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def as2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return a.reshape(1, -1), True
        return a, False

    # Convert inputs
    x, x_was_1d = as2d(x)
    h_prev, h_was_1d = as2d(h_prev)

    # Extract parameters
    Wz = np.asarray(params["Wz"], dtype=float)
    Uz = np.asarray(params["Uz"], dtype=float)
    bz = np.asarray(params["bz"], dtype=float)

    Wr = np.asarray(params["Wr"], dtype=float)
    Ur = np.asarray(params["Ur"], dtype=float)
    br = np.asarray(params["br"], dtype=float)

    Wh = np.asarray(params["Wh"], dtype=float)
    Uh = np.asarray(params["Uh"], dtype=float)
    bh = np.asarray(params["bh"], dtype=float)

    # Gates
    z = sigmoid(x @ Wz + h_prev @ Uz + bz)
    r = sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)

    # Final state
    h_t = (1 - z) * h_prev + z * h_tilde

    # Convert back to 1D if needed
    if x_was_1d:
        h_t = h_t.reshape(-1)

    return h_t
