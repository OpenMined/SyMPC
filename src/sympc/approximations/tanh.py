"""function used to calculate tanh of a given tensor."""

from sympc.approximations.sigmoid import sigmoid


def _tanh_sigmoid(tensor):
    """Compute the tanh using the sigmoid approximation.

    Args:
        tensor (tensor): values where tanh should be approximated

    Returns:
        tensor (tensor): tanh calculated using sigmoid
    """
    return 2 * sigmoid(2 * tensor) - 1


def tanh(tensor, method="sigmoid"):
    """Calculates tanh of given tensor.

    Args:
        tensor: whose sigmoid has to be calculated
        method: method to use while calculating sigmoid

    Returns:
        MPCTensor: calculated MPCTensor

    Raises:
        ValueError: if the requested method does not exists.

    """
    if method == "sigmoid":
        return _tanh_sigmoid(tensor)
    else:
        raise ValueError("The request method does not exists")
