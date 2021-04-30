"""function used to calculate sigmoid of a given tensor."""

# third party
import torch

from sympc.approximations.exponential import exp
from sympc.approximations.reci import reciprocal
from sympc.approximations.utils import sign
from sympc.tensor.mpc_tensor import MPCTensor


def sigmoid(tensor: MPCTensor, method: str = "exp") -> "MPCTensor":
    """Approximates the sigmoid function using a given method.

    Args:
        tensor: tensor to calculate sigmoid
        method (str): (default = "chebyshev")
            Possible values: "exp", "maclaurin", "chebyshev"

    Returns:
        tensor: the calulated sigmoid value

    Raises:
        ValueError: if the given method is not supported
    """
    if method == "exp":
        _sign = sign(tensor)

        # Make sure the elements are all positive
        x = tensor * _sign
        ones = tensor * 0 + 1
        half = ones / 2
        result = reciprocal(ones + exp(-1 * ones * x), method="nr")
        return (result - half) * _sign + half

    elif method == "maclaurin":
        weights = torch.tensor([0.5, 1.91204779e-01, -4.58667307e-03, 4.20690803e-05])
        degrees = [0, 1, 3, 5]

        # initiate with term of degree 0 to avoid errors with tensor ** 0
        one = tensor * 0 + 1
        result = one * weights[0]
        for i, d in enumerate(degrees[1:]):
            result += (tensor ** d) * weights[i + 1]

        return result

    elif method == "chebyshev":
        # Reference: http://www.nnw.cz/doi/2012/NNW.2012.22.023.pdf
        # Make sure the elements are all positive
        _sign = sign(tensor)
        tensor_8 = tensor * _sign / 8
        p = 11
        q = 11
        scaler = ((1 + tensor_8) / 2) ** (q + 1)

        def factorial(n):
            fact = 1
            for i in range(1, n + 1):
                fact = fact * i
            return fact

        polynomial = 0 * tensor_8
        for mu in range(p + 1):
            a_n = factorial(mu + q) / (factorial(mu) * factorial(q))
            T_n_w = ((1 - tensor_8) / 2) ** mu
            polynomial += a_n * T_n_w

        result = scaler * polynomial

        return ((1 - _sign) * (1 - result) + (1 + _sign) * (result)) / 2

    else:
        raise ValueError(f"Invalid method {method} given for sigmoid function")
