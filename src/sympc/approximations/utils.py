from sympc.approximations.exponential import exp
from sympc.approximations.log import log


def sign(data):
    return (data > 0) + (data < 0) * (-1)


def modulus(data):
    """
    Calculation of modulus for a given tensor
    """
    return data.signum() * data


def signum(data):
    """
    Calculation of signum function for a given tensor
    """
    sgn = (data > 0) - (data < 0)
    return sgn


def reciprocal(data, method="NR", nr_iters=10):
    r"""
    Calculate the reciprocal using the algorithm specified in the method args.
    Ref: https://github.com/facebookresearch/CrypTen
    Args:
        method:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default
        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`
        nr_iters:
            Number of iterations for `Newton-Raphson`
    Returns:
        Reciprocal of `self`
    """
    method = method.lower()

    if method == "nr":
        new_data = modulus(data)
        result = 3 * exp(0.5 - new_data) + 0.003
        for i in range(nr_iters):
            result = 2 * result - result * result * new_data
        return result * signum(data)
    elif method == "newton":
        # Note: this computes the SQRT of the reciprocal !!
        # it is assumed here that input values are taken in [-20, 20]
        x = None
        C = 20
        for i in range(80):
            if x is not None:
                y = C + 1 - data * (x * x)
                x = y * x / C
            else:
                y = C + 1 - data
                x = y / C
        return x
    elif method == "division":
        ones = data * 0 + 1
        return ones / data
    elif method == "log":
        new_data = modulus(data)
        return exp(log(-new_data)) * signum(data)
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")
