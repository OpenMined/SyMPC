import torch
from copy import deepcopy

class Beaver:
    @staticmethod
    def build_triples(op, x, y):
        """
        The Trusted Third Party (TTP) or Crypto Provider should provide this triples
        Currently, the one that orchestrates the communication provides those
        """
        session = x.session
        shape_x = x.shape
        shape_y = y.shape
        min_val = 10#session.conf.min_value
        max_val = 11#session.conf.max_value

        a = torch.randint(min_val, max_val, shape_x).long()
        b = torch.randint(min_val, max_val, shape_y).long()

        cmd = getattr(torch, op)
        c = cmd(a, b)

        import pdb; pdb.set_trace()
        from sympc.tensor import AdditiveSharingTensor

        a_sh = AdditiveSharingTensor(secret=a, session=session)
        b_sh = AdditiveSharingTensor(secret=b, session=session)
        c_sh = AdditiveSharingTensor(secret=c, session=session)

        return a_sh, b_sh, c_sh
