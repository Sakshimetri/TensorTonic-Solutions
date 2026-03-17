import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    Returns (new_w, new_m, new_v)
    """

    # Ensure numpy arrays
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: update first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # Step 2: update second moment
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: decoupled weight decay + Adam update
    w_new = w - lr * (weight_decay * w) - lr * (m_new / (np.sqrt(v_new) + eps))

    return w_new, m_new, v_new