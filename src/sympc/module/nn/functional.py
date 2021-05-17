"""Implementations for torch.nn.functional equivalent for MPC."""

from sympc.tensor import MPCTensor


def relu(x: MPCTensor) -> MPCTensor:
    """Rectified linear unit function.

    Args:
        x (MPCTensor): The tensor on which we apply the function

    Returns:
        An MPCTensor which represents the ReLu applied on the input tensor
    """
    res = x * (x >= 0)
    return res


def mse_loss(pred: MPCTensor, target: MPCTensor) -> MPCTensor:
    """Mean Squared Error loss.

    Args:
        pred (MPCTensor): The predictions obtained
        target (MPCTensor): The target values

    Returns:
        The loss
    """
    return (pred - target).pow(2).sum()
