from sympc.approximations.sigmoid import sigmoid


def _tanh_sigmoid(tensor):
    """
    Compute the tanh using the sigmoid approximation
    Args:
        tensor (tensor): values where tanh should be approximated
    """

    return 2 * sigmoid(2 * tensor) - 1


def tanh(tensor, method="sigmoid"):
    if method == "sigmoid":
        return _tanh_sigmoid(tensor)
    else:
        raise ValueError("The request method does not exists")
