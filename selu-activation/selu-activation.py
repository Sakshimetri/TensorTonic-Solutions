import math

def selu(x):
    # Exact constants (must use these values)
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    
    # Ensure we iterate element-wise (input guaranteed list)
    result = []
    for val in x:
        if val > 0:
            result.append(lam * val)
        else:
            result.append(lam * alpha * (math.exp(val) - 1))
    
    return result