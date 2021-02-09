def exp(value, iterations=8):
    r"""
    Approximates the exponential function using a limit approximation:
    exp(x) = \lim_{n -> infty} (1 + x / n) ^ n
    Here we compute exp by choosing n = 2 ** d for some large d equal to
    iterations. We then compute (1 + x / n) once and square `d` times.
    Args:
        iterations (int): number of iterations for limit approximation
    Ref: https://github.com/LaRiffle/approximate-models
    """
    return (1 + value / 2 ** iterations) ** (2 ** iterations)
