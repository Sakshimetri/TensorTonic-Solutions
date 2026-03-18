import math

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """

    # Compute Xavier limit
    limit = math.sqrt(6.0 / (fan_in + fan_out))

    # Map [0,1] → [-limit, limit]
    W_scaled = [
        [(w * 2 * limit) - limit for w in row]
        for row in W
    ]

    return W_scaled