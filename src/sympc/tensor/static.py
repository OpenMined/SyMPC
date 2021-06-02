""" TODO: Fill me """
# third party
import torch

from sympc.tensor.share_tensor import ShareTensor
from sympc.utils import parallel_execution


def stack(tensors, dim=0):
    """ TODO: Fill me """
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
    """ # TODO: Fill me """
    result = ShareTensor(session=shares[0].session)
    result.tensor = torch.stack([share.tensor for share in shares])
    return result


def cat():
    """ TODO: Fill me """
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
    """ # TODO: Fill me """
    result = ShareTensor(session=shares[0].session)
    result.tensor = torch.cat([share.tensor for share in shares])
    return result
