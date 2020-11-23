import torch
import operator
from copy import deepcopy

from ...tensor import modulo

EXPECTED_OPS = {"matmul", "mul"}

def build_triples(x, y, op_str):
    """
    The Trusted Third Party (TTP) or Crypto Provider should provide this triples
    Currently, the one that orchestrates the communication provides those
    """
    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    session = x.session
    shape_x = x.shape
    shape_y = y.shape
    conf = session.config
    min_val = conf.min_value
    max_val = conf.max_value

    # TODO: Move this to a library specific file
    a = torch.randint(min_val, max_val, shape_x).long()
    b = torch.randint(min_val, max_val, shape_y).long()

    cmd = getattr(operator, op_str)
    c = modulo(cmd(a, b).long(), session)

    from sympc.tensor import AdditiveSharingTensor

    session_copy = session.get_copy()
    session_copy.config.enc_precision = 0

    a_sh = AdditiveSharingTensor(secret=a, session=session_copy)
    b_sh = AdditiveSharingTensor(secret=b, session=session_copy)
    c_sh = AdditiveSharingTensor(secret=c, session=session_copy)

    return a_sh, b_sh, c_sh
