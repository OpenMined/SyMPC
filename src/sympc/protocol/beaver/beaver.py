from typing import Tuple

import torch
import operator

from sympc.tensor.share_control import ShareTensorCC
from sympc.tensor.share import ShareTensor

EXPECTED_OPS = {"matmul", "mul"}


def build_triples(
    x: ShareTensorCC, y: ShareTensorCC, op_str: str
) -> Tuple[ShareTensorCC, ShareTensorCC, ShareTensorCC]:

    """
    The Trusted Third Party (TTP) or Crypto Provider should provide this triples
    Currently, the one that orchestrates the communication provides those
    """
    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    shape_x = x.shape
    shape_y = y.shape

    session = x.session
    min_val = session.min_value
    max_val = session.max_value

    a = ShareTensor(session=session)
    a.tensor = torch.randint(min_val, max_val, shape_x, dtype=torch.long)

    b = ShareTensor(session=session)
    b.tensor = torch.randint(min_val, max_val, shape_y, dtype=torch.long)

    cmd = getattr(operator, op_str)

    # Manually place the tensor and use the same session such that we do
    # not encode the result
    c = ShareTensor(session=session)
    c.tensor = cmd(a.tensor, b.tensor)

    a_sh = ShareTensorCC(secret=a, session=session)
    b_sh = ShareTensorCC(secret=b, session=session)
    c_sh = ShareTensorCC(secret=c, session=session)

    return a_sh, b_sh, c_sh
