from typing import List

from itertools import repeat

import torch
import operator
from sympc.tensor import ShareTensor
from concurrent.futures import ThreadPoolExecutor, wait

from sympc.session import Session
from sympc.protocol import beaver
from sympc.tensor import ShareTensor
from sympc.tensor import ShareTensorCC
from sympc.utils import parallel_execution


EXPECTED_OPS = {"mul", "matmul"}


""" Functions that are executed at the orchestrator """


def mul_master(x: ShareTensorCC, y: ShareTensorCC, op_str: str) -> List[ShareTensor]:

    """
    [c] = [a * b]
    [eps] = [x] - [a]
    [delta] = [y] - [b]

    Open eps and delta
    [result] = [c] + eps * [b] + delta * [a] + eps * delta
    """

    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    a_sh, b_sh, c_sh = beaver.build_triples(x, y, op_str)
    eps = x - a_sh
    delta = y - b_sh
    session = x.session
    nr_parties = len(session.session_ptr)

    eps_plaintext = eps.reconstruct(decode=False)
    delta_plaintext = delta.reconstruct(decode=False)

    args = list(
        map(
            list,
            zip(session.session_ptr, a_sh.share_ptrs, b_sh.share_ptrs, c_sh.share_ptrs),
        )
    )

    for i in range(nr_parties):
        args[i].extend([eps_plaintext, delta_plaintext, op_str])

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

    op = getattr(operator, op_str)

    eps_b = op(eps, b_share)
    delta_a = op(delta, a_share)

    share = c_share + eps_b + delta_a
    if session.rank == 0:
        delta_eps = op(delta, eps)
        share.tensor = share.tensor + delta_eps

    scale = session.config.encoder_base ** session.config.encoder_precision
    share.tensor //= scale

    # Convert to our tensor type
    share.tensor = share.tensor.type(session.tensor_type)

    return share
