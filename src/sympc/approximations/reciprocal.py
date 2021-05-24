"""function used to calculate reciprocal of a given tensor."""

from sympc.approximations.exponential import exp
from sympc.approximations.log import log
from sympc.approximations.utils import modulus
from sympc.approximations.utils import sign


def reciprocal(self, method: str = "NR", nr_iters: int = 10):
    r"""Calculate the reciprocal using the algorithm specified in the method args.

    Ref: https://github.com/facebookresearch/CrypTen

    Args:
        self: input data
        nr_iters: Number of iterations for Newton-Raphson
        method: 'NR' - `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(-(x-.5)) + 0.003` as an initial guess by default

                'log' -  Computes the reciprocal of the input from the observation that:
                        :math:`x^{-1} = exp(-log(x))`

    Returns:
        Reciprocal of `self`

    Raises:
        ValueError: if the given method is not supported
    """
    method = method.lower()

    if method == "nr":
        new_self = modulus(self)
        result = 3 * exp(0.5 - new_self) + 0.003
        for i in range(nr_iters):
            result = 2 * result - result * result * new_self
        return result * sign(self)
    elif method == "log":
        new_self = modulus(self)
        return exp(-1 * log(new_self)) * sign(self)
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")
