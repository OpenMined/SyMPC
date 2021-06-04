"""Module level "static" functions.

We use this to support torch.method(Tensor)
"""
# future
from __future__ import annotations

# stdlib
from typing import List
from typing import TYPE_CHECKING
from typing import Tuple
from uuid import UUID

# third party
import torch

from sympc.session import get_session
from sympc.tensor.share_tensor import ShareTensor
from sympc.utils import parallel_execution

if TYPE_CHECKING:
    from sympc.tensor import MPCTensor


def stack(tensors: List, dim: int = 0) -> MPCTensor:
    """Concatenates a sequence of tensors along a new dimension.

    Args:
        tensors (List): sequence of tensors to stacks
        dim (int): dimension to insert. Has to be between 0 and the number of
            dimensions of concatenated tensors (inclusive)

    Returns:
        MPCTensor: calculated MPCTensor
    """
    session = tensors[0].session

    args = list(
        zip(
            [str(uuid) for uuid in session.rank_to_uuid.values()],
            *[tensor.share_ptrs for tensor in tensors]
        )
    )

    stack_shares = parallel_execution(stack_share_tensor, session.parties)(args)
    from sympc.tensor import MPCTensor

    expected_shape = torch.stack(
        [torch.empty(each_tensor.shape) for each_tensor in tensors], dim=dim
    ).shape
    result = MPCTensor(shares=stack_shares, session=session, shape=expected_shape)

    return result


def stack_share_tensor(
    session_uuid_str: str, *shares: Tuple[ShareTensor]
) -> ShareTensor:
    """Helper method that performs torch.stack on the shares of the Tensors.

    Args:
        session_uuid_str (str): UUID to identify the session on each party side.
        shares (Tuple[ShareTensor]) : Shares of the tensors to be stacked

    Returns:
        ShareTensor: Respective shares after stacking
    """
    session = get_session(session_uuid_str)
    result = ShareTensor(session_uuid=UUID(session_uuid_str), config=session.config)

    result.tensor = torch.stack([share.tensor for share in shares])
    return result


def cat(tensors: List, dim: int = 0) -> MPCTensor:
    """Concatenates the given sequence of seq tensors in the given dimension.

    Args:
        tensors (List): sequence of tensors to concatenate
        dim (int): the dimension over which the tensors are concatenated

    Returns:
        MPCTensor: calculated MPCTensor
    """
    session = tensors[0].session

    args = list(
        zip(
            [str(uuid) for uuid in session.rank_to_uuid.values()],
            *[tensor.share_ptrs for tensor in tensors]
        )
    )

    stack_shares = parallel_execution(cat_share_tensor, session.parties)(args)
    from sympc.tensor import MPCTensor

    expected_shape = torch.cat(
        [torch.empty(each_tensor.shape) for each_tensor in tensors], dim=dim
    ).shape
    result = MPCTensor(shares=stack_shares, session=session, shape=expected_shape)

    return result


def cat_share_tensor(session_uuid_str: str, *shares: Tuple[ShareTensor]) -> ShareTensor:
    """Helper method that performs torch.cat on the shares of the Tensors.

    Args:
        session_uuid_str (str): UUID to identify the session on each party side.
        shares (Tuple[ShareTensor]): Shares of the tensors to be concatenated

    Returns:
        ShareTensor: Respective shares after concatenation
    """
    session = get_session(session_uuid_str)
    result = ShareTensor(session_uuid=UUID(session_uuid_str), config=session.config)

    result.tensor = torch.cat([share.tensor for share in shares])
    return result


STATIC_FUNCS = {"stack": stack, "cat": cat}
