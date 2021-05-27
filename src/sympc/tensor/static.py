""" TODO: Fill me """
# third party
import torch

from sympc.utils import parallel_execution


def stack(tensors, dim=0):
    """ TODO: Fill me """
    session = tensors[0].session
    share_ptrs = [tensor.share_ptrs for tensor in tensors]
    args = share_ptrs

    stack_shares = parallel_execution(_stack_share_tensor, session.parties)(args)
    from sympc.tensor import MPCTensor

    result = MPCTensor(shares=stack_shares, session=session)

    return result


def _stack_share_tensor(*shares):
    """ # TODO: Fill me """
    result = torch.stack([*shares])
    return result


def cat():
    """ TODO: Fill me """
