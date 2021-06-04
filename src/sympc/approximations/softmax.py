"""function used to calculate softmax and its variants for a given tensor."""

from sympc.approximations.exponential import exp
from sympc.approximations.log import log
from sympc.approximations.reciprocal import reciprocal
from sympc.tensor import MPCTensor


def softmax(tensor: MPCTensor, dim: int = 0) -> MPCTensor:
    """Calculates tanh of given tensor's elements along the given dimension.

    Args:
        tensor: whose softmax has to be calculated
        dim: dim along which softmax is to be calculated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    # Single Element along dim
    if tensor.shape[dim] == 1:
        return 0 * tensor + 1  # Equivalent to torch.ones_like(tensor)

    # Waiting for https://github.com/OpenMined/SyMPC/pull/173/
    maximum_value = tensor.max(dim, keepdim=True)[0]
    logits = tensor - maximum_value
    numerator = logits.exp()

    # Sum not implemented yet
    denominator = numerator.sum(dim, keepdim=True)
    return numerator * reciprocal(denominator)


def log_softmax(tensor: MPCTensor, dim: int = 0) -> MPCTensor:
    """Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    Args:
        tensor: whose log-softmax has to be calculated
        dim: dim along which log-softmax is to be calculated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    if tensor.size(dim) == 1:
        return 0 * tensor + 1  # Equivalent to torch.ones_like(tensor)

    # Waiting for https://github.com/OpenMined/SyMPC/pull/173/
    maximum_value = tensor.max(dim, keepdim=True)[0]
    logits = tensor - maximum_value

    # Sum not implemented yet
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - log(normalize_term)
    return result
