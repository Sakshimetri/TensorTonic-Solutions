import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # Step 1: accumulate
    G_new = G + g**2

    # Step 2: correct formula (eps INSIDE sqrt)
    w_new = w - lr * g / np.sqrt(G_new + eps)

    return w_new, G_new