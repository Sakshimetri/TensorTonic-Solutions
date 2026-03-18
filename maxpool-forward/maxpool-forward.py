def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    Returns: list of lists
    """

    H = len(X)
    W = len(X[0])

    # Output dimensions
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = []

    for i in range(H_out):
        row = []
        for j in range(W_out):
            max_val = float('-inf')

            # Pooling window
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * stride + a][j * stride + b]
                    if val > max_val:
                        max_val = val

            row.append(max_val)
        out.append(row)

    return out