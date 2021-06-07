"""function used to calculate softmax and its variants for a given tensor."""

# stdlib
from typing import Optional

from sympc.approximations.exponential import exp
from sympc.approximations.log import log
from sympc.approximations.reciprocal import reciprocal
from sympc.tensor import MPCTensor


def softmax(tensor: MPCTensor, dim: Optional[int] = None) -> MPCTensor:
    """Calculates tanh of given tensor's elements along the given dimension.

    Args:
        tensor (MPCTensor): whose softmax has to be calculated
        dim (int): dim along which softmax is to be calculated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    if dim is None:
        dim = len(tensor.shape) - 1

    # Single Element along dim
    if tensor.shape[dim] == 1:
        przs = MPCTensor.generate_przs(shape=tensor.shape, session=tensor.session)
        zeros = MPCTensor(tensor.session, shape=tensor.shape, shares=przs)
        return zeros + 1  # Equivalent to torch.ones_like(tensor)

    maximum_value = tensor.max(dim, keepdim=True)[0]
    logits = tensor - maximum_value
    numerator = exp(logits)

    denominator = numerator.sum(dim, keepdim=True)
    return numerator * reciprocal(denominator)


def log_softmax(tensor: MPCTensor, dim: Optional[int] = None) -> MPCTensor:
    """Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    Args:
        tensor (MPCTensor): whose log-softmax has to be calculated
        dim (int): dim along which log-softmax is to be calculated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    if dim is None:
        dim = len(tensor.shape) - 1

    # Single Element along dim
    if tensor.shape[dim] == 1:
        przs = MPCTensor.generate_przs(shape=tensor.shape, session=tensor.session)
        zeros = MPCTensor(tensor.session, shape=tensor.shape, shares=przs)
        return zeros  # Equivalent to torch.zeros_like(tensor)

    maximum_value = tensor.max(dim, keepdim=True)[0]
    logits = tensor - maximum_value

    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - log(normalize_term)
    return result
