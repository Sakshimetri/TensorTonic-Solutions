def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    Returns a list (descent direction).
    """

    # Number of history pairs
    m = len(s_list)

    # Step 1: compute rho values
    rho = [1.0 / _dot(y_list[i], s_list[i]) for i in range(m)]

    # Step 2: backward loop
    q = grad[:]  # copy
    alpha = [0.0] * m

    for i in reversed(range(m)):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [q[j] - alpha[i] * y_list[i][j] for j in range(len(q))]

    # Step 3: initial Hessian scaling (gamma)
    s_last = s_list[-1]
    y_last = y_list[-1]
    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)

    # Step 4: apply scaling
    r = [gamma * qi for qi in q]

    # Step 5: forward loop
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [
            r[j] + s_list[i][j] * (alpha[i] - beta)
            for j in range(len(r))
        ]

    # Step 6: return descent direction
    direction = [-ri for ri in r]
    return direction
