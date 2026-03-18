import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """

    # Compute limit
    limit = math.sqrt(6.0 / fan_in)

    # Scale weights from [0,1] → [-limit, limit]
    W_scaled = [
        [(w * 2 * limit) - limit for w in row]
        for row in W
    ]

    return W_scaled