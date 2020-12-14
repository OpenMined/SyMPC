"""
The Beaver Triples
"""

from typing import Tuple

import torch
import torchcsprng as csprng  # type: ignore
import operator

from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.util import count_wraps

EXPECTED_OPS = {"matmul", "mul"}

ttp_generator = csprng.create_random_device_generator()

""" Those functions should be executed by the Trusted Party """

def build_triples(
    x: MPCTensor, y: MPCTensor, op_str: str
) -> Tuple[MPCTensor, MPCTensor, MPCTensor]:

    """
    The Trusted Third Party (TTP) or Crypto Provider should provide this triples
    Currently, the one that orchestrates the communication provides those
    """
    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    shape_x = x.shape
    shape_y = y.shape

    session = x.session

    a = ShareTensor(session=session)
    a.tensor = torch.zeros(shape_x, dtype=torch.long)

    b = ShareTensor(session=session)
    b.tensor = torch.zeros(shape_y, dtype=torch.long)

    cmd = getattr(operator, op_str)

    # Manually place the tensor and use the same session such that we do
    # not encode the result
    c = ShareTensor(session=session)
    c.tensor = cmd(a.tensor, b.tensor)

    a_sh = MPCTensor(secret=a, session=session)
    b_sh = MPCTensor(secret=b, session=session)
    c_sh = MPCTensor(secret=c, session=session)

    return a_sh, b_sh, c_sh


def count_wraps_rand(x: MPCTensor):
    shape_x = x.shape

    session = x.session

    r = ShareTensor(session=session)
    r._tensor = torch.zeors(shape_x, dtype=torch.long)

    shares = MPCTensor.generate_shares(r, session)

    theta_r = count_wraps(shares)

    r = MPCTensor(shares=shares, session=session)
    theta_r = MPCTensor(secret=theta_r, session=session)

    return r, theta_r
