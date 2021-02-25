from typing import List

from sympc.protocol import Protocol
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import decompose
from sympc.utils import parallel_execution

import torch


class ABY3(metaclass=Protocol):
    @staticmethod
    def A2B(x: MPCTensor) -> List[MPCTensor]:
        """
        Convert an Arithmetic Shared Value to a Binary Shared Value
        (from ring size 2^(pow != 1) to ring size 2)
        """

        session = x.session
        parties = session.parties
        ring_size = session.ring_size
        shape = x.shape

        x_reshared = [
            MPCTensor(secret=share, session=x.session, shape=x.shape)
            for share in x.share_ptrs
        ]

        args = [
            [x_reshared[0].share_ptrs[0], x_reshared[1].share_ptrs[0]],
            [x_reshared[0].share_ptrs[1], x_reshared[1].share_ptrs[1]],
        ]

        p = parallel_execution(ABY3.sum_shares, session.parties)(args)
        import pdb

        pdb.set_trace()
        # session.ring_size = 2
        # x_share_binary = [MPCTensor(session=session, secret=x_share)]
        # MPCTensor.generate_przs(x.shape, session)

    def sum_shares(*shares: List[ShareTensor]) -> List[ShareTensor]:
        session = shares[0].session
        res = ShareTensor(session=session)

        x = torch.stack([shareTensor.data for shareTensor in shares])

        # Add all BinarySharedTensors
        while x.size(0) > 1:
            extra = None
            if x.size(0) % 2 == 1:
                extra = x[0]
                x = x[1:]
            x0 = x[: (x.size(0) // 2)]
            x1 = x[(x.size(0) // 2) :]
            x = x0 + x1
            if extra is not None:
                x = torch_cat([x.share, extra.unsqueeze(0)])

        res.tensor = x
        return res
