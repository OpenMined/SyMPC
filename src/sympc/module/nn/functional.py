"""Implementations for torch.nn.functional equivalent for MPC."""

from sympc.tensor import MPCTensor


def relu(x: MPCTensor) -> MPCTensor:
    """Rectified linear unit function.

    Args:
        x (MPCTensor): The tensor on which we apply the function to

    Returns:
        An MPCTensor which represents the ReLu applied on the input tensor
    """
    res = x * (x >= 0)
    return res
