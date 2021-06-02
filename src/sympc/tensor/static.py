"""Module level "static" functions.

We use this to support torch.method(Tensor)
"""
# third party
import torch

from sympc.tensor.share_tensor import ShareTensor
from sympc.utils import parallel_execution


def stack(tensors, dim=0):
    """Concatenates a sequence of tensors along a new dimension.

    Args:
        tensors: sequence of tensors to stacks
        dim: dimension to insert. Has to be between 0 and the number of
            dimensions of concatenated tensors (inclusive)

    Returns:
        MPCTensor: calculated MPCTensor
    """
    session = tensors[0].session

    # Each MPCTensor has
    # share_1, share_2 owned by
    # Party1    Party2

    share_ptrs = list(zip(*[tensor.share_ptrs for tensor in tensors]))
    args = share_ptrs

    stack_shares = parallel_execution(stack_share_tensor, session.parties)(args)
    from sympc.tensor import MPCTensor

    expected_shape = torch.stack(
        [torch.empty(each_tensor.shape) for each_tensor in tensors], dim=dim
    ).shape
    result = MPCTensor(shares=stack_shares, session=session, shape=expected_shape)

    return result


def stack_share_tensor(*shares):
    """Helper method that performs torch.stack on the shares of the Tensors.

    Args:
        shares: Shares of the tensors to be stacked

    Returns:
        ShareTensorPointer: Respective shares after stacking
    """
    result = ShareTensor(session=shares[0].session)
    result.tensor = torch.stack([share.tensor for share in shares])
    return result


def cat(tensors, dim=0):
    """Concatenates the given sequence of seq tensors in the given dimension.

    Args:
        tensors: sequence of tensors to concatenate
        dim: the dimension over which the tensors are concatenated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    session = tensors[0].session

    # Each MPCTensor has
    # share_1, share_2 owned by
    # Party1    Party2

    share_ptrs = list(zip(*[tensor.share_ptrs for tensor in tensors]))
    args = share_ptrs

    stack_shares = parallel_execution(cat_share_tensor, session.parties)(args)
    from sympc.tensor import MPCTensor

    expected_shape = torch.cat(
        [torch.empty(each_tensor.shape) for each_tensor in tensors], dim=dim
    ).shape
    result = MPCTensor(shares=stack_shares, session=session, shape=expected_shape)

    return result


def cat_share_tensor(*shares):
    """Helper method that performs torch.cat on the shares of the Tensors.

    Args:
        shares: Shares of the tensors to be concatenated

    Returns:
        ShareTensorPointer: Respective shares after concatenation
    """
    result = ShareTensor(session=shares[0].session)
    result.tensor = torch.cat([share.tensor for share in shares])
    return result


STATIC_FUNCS = {"stack": stack, "cat": cat}
