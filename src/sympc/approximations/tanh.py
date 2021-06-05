"""function used to calculate tanh of a given tensor."""

# stdlib
import functools
from typing import Callable

# third party
import numpy as np
import torch
from torch import Tensor

from sympc.approximations.sigmoid import sigmoid
from sympc.module.nn import relu
from sympc.tensor import MPCTensor
from sympc.tensor.static import stack


def _tanh_sigmoid(tensor):
    """Compute the tanh using the sigmoid approximation.

    Args:
        tensor (tensor): values where tanh should be approximated

    Returns:
        tensor (tensor): tanh calculated using sigmoid
    """
    return 2 * sigmoid(2 * tensor) - 1


def tanh(tensor: MPCTensor, method: str = "sigmoid") -> MPCTensor:
    """Calculates tanh of given tensor.

    Args:
        tensor (MPCTensor): whose sigmoid has to be calculated
        method (str): method to use while calculating sigmoid

    Returns:
        MPCTensor: calculated MPCTensor

    Raises:
        ValueError: if the requested method does not exists.

    """
    if method == "sigmoid":
        return _tanh_sigmoid(tensor)

    elif method == "chebyshev":
        terms = 10  # Higher terms gives slower but more accurate results
        coeffs = chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(tensor, terms)
        tanh_polys_flipped = tanh_polys.unsqueeze(dim=-1).T.squeeze(dim=0)
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return hardtanh(out)
    else:
        raise ValueError(f"Invalid method {method} given for tanh function")


def hardtanh(
    tensor: MPCTensor, min_value: float = -1, max_value: float = 1
) -> MPCTensor:
    """Calculates hardtanh of given tensor.

    Defined as
        1 if x > 1
        -1 if x < -1
        x otherwise

    Args:
        tensor (MPCTensor): whose hardtanh has to be calculated
        min_value (float): minimum value of the linear region range. Default: -1
        max_value (float): maximum value of the linear region range. Default: 1

    Returns:
        MPCTensor: calculated MPCTensor
    """
    intermediate = relu(tensor - min_value) - relu(tensor - max_value)
    return intermediate + min_value


@functools.lru_cache(maxsize=10)
def chebyshev_series(func: Callable, width: int, terms: int) -> Tensor:
    r"""Computes Chebyshev coefficients.

    For n = terms, the ith Chebyshev series coefficient is

        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))

    Args:
        func (Callable): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation

    Returns:
        Tensor: Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = torch.arange(start=0, end=terms).float()
    x = width * torch.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = torch.cos(torch.ger(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * torch.sum(y * cos_term, axis=1)
    return coeffs


def _chebyshev_polynomials(tensor: MPCTensor, terms: int) -> MPCTensor:
    r"""Evaluates odd degree Chebyshev polynomials at x.

    Chebyshev Polynomials of the first kind are defined as
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        tensor (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials. Must be even and at least 6.

    Returns:
        MPCTensor: polynomials evaluated at self of shape `(terms, *self)`

    Raises:
        ValueError: if terms < 6 or is not divisible by 2
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [tensor.clone()]
    y = 4 * tensor * tensor - 2
    z = y - 1
    polynomials.append(z * tensor)

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return stack(polynomials)
