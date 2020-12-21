"""
SPDZ mechanism used for multiplication
Contains functions that are run at:
* the party that orchestrates the computation
* the parties that hold the shares
"""

import operator
from typing import List

import torch

from sympc.protocol.beaver import beaver
from sympc.session import Session
from sympc.tensor import MPCTensor, ShareTensor
from sympc.utils import parallel_execution

EXPECTED_OPS = {"mul", "matmul"}


""" Functions that are executed at the orchestrator """


def mul_master(x: MPCTensor, y: MPCTensor, op_str: str) -> List[ShareTensor]:
    """Function that is executed by the orchestrator to multiply two secret values

    :return: a new set of shares that represents the multiplication
           between two secret values
    :rtype: MPCTensor
    """

    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    a_sh, b_sh, c_sh = beaver.build_triples(x, y, op_str)
    eps = x - a_sh
    delta = y - b_sh
    session = x.session
    nr_parties = len(session.session_ptrs)

    eps_plaintext = eps.reconstruct(decode=False)
    delta_plaintext = delta.reconstruct(decode=False)

    # Arguments that must be sent to all parties
    common_args = [eps_plaintext, delta_plaintext, op_str]

    # Specific arguments to each party
    args = zip(session.session_ptrs, a_sh.share_ptrs, b_sh.share_ptrs, c_sh.share_ptrs)
    args = [list(el) + common_args for el in args]

    shares = parallel_execution(mul_parties, session.parties)(args)
    return shares


""" Functions that are executed at each party that holds shares """


def mul_parties(
    session: Session,
    a_share: ShareTensor,
    b_share: ShareTensor,
    c_share: ShareTensor,
    eps: torch.Tensor,
    delta: torch.Tensor,
    op_str: str,
) -> ShareTensor:
    """
    [c] = [a * b]
    [eps] = [x] - [a]
    [delta] = [y] - [b]

    Open eps and delta
    [result] = [c] + eps * [b] + delta * [a] + eps * delta

    :return: the ShareTensor for the multiplication
    :rtype: ShareTensor (in our case ShareTensorPointer)
    """

    op = getattr(operator, op_str)

    eps_b = op(eps, b_share.tensor)
    delta_a = op(a_share.tensor, delta)

    share_tensor = c_share.tensor + eps_b + delta_a
    if session.rank == 0:
        delta_eps = op(eps, delta)
        share_tensor += delta_eps

    scale = session.config.encoder_base ** session.config.encoder_precision

    share_tensor //= scale

    # Convert to our tensor type
    share_tensor = share_tensor.type(session.tensor_type)

    share = ShareTensor(session=session)
    share.tensor = share_tensor

    return share
