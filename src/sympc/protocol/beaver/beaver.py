import torch

def build_triples(op, x, y):
    """
    The Trusted Third Party (TTP) or Crypto Provider should provide this triples
    Currently, the one that orchestrates the communication provides those
    """
    session = x.session
    shape_x = x.shape
    shape_y = y.shape
    min_val = session.conf.min_value
    max_val = session.conf.max_value

    a = torch.randint(min_val, max_val, shape_x).long()
    b = torch.randint(min_val, max_val, shape_y).long()

    cmd = getattr(torch, op)
    c = cmd(a, b)

    from sympc.tensor import AdditiveSharingTensor

    kwargs = {
        "session": session,
        "encoder_precision": 0
    }
    a_sh = AdditiveSharingTensor(secret=a, **kwargs)
    b_sh = AdditiveSharingTensor(secret=b, **kwargs)
    c_sh = AdditiveSharingTensor(secret=c, **kwargs)

    return a_sh, b_sh, c_sh
