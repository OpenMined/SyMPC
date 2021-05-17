"""function used to calculate sigmoid of a given tensor."""
# stdlib
from typing import Any

# third party
import torch

from sympc.approximations.exponential import exp
from sympc.approximations.reciprocal import reciprocal
from sympc.approximations.utils import sign


def sigmoid(tensor: Any, method: str = "exp") -> Any:
    """Approximates the sigmoid function using a given method.

    Args:
        tensor (Any): tensor to calculate sigmoid
        method (str): (default = "chebyshev")
            Possible values: "exp", "maclaurin", "chebyshev"

    Returns:
        tensor (Any): the calulated sigmoid value
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
