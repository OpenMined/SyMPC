from typing import Tuple

import torch
import torchcsprng as csprng
import operator

from sympc.tensor.share_control import ShareTensorCC
from sympc.tensor.share import ShareTensor

EXPECTED_OPS = {"matmul", "mul"}

ttp_generator = csprng.create_random_device_generator()


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

    a = ShareTensor(session=session)
    a.tensor = torch.empty(size=shape_x, dtype=torch.long).random_(
        generator=ttp_generator
    )

    b = ShareTensor(session=session)
    b.tensor = torch.empty(size=shape_x, dtype=torch.long).random_(
        generator=ttp_generator
    )

    cmd = getattr(operator, op_str)

    # Manually place the tensor and use the same session such that we do
    # not encode the result
    c = ShareTensor(session=session)
    c.tensor = cmd(a.tensor, b.tensor)

    a_sh = ShareTensorCC(secret=a, session=session)
    b_sh = ShareTensorCC(secret=b, session=session)
    c_sh = ShareTensorCC(secret=c, session=session)

    return a_sh, b_sh, c_sh
