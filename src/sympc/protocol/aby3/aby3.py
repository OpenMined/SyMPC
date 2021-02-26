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
        To preserve space each Binary Shared Value is kept on a bit
        """

        session = x.session
        parties = session.parties
        nr_parties = session.nr_parties
        ring_size = session.ring_size
        shape = x.shape

        x_reshared = [
            MPCTensor(secret=share, session=session, mpc_type="binary", shape=x.shape)
            for share in x.share_ptrs
        ]

        print("VALUES")
        print((x.share_ptrs[0] + 0).get())
        print((x.share_ptrs[1] + 0).get())
        print(x_reshared[0].reconstruct(decode=False))
        print(x_reshared[1].reconstruct(decode=False))

        args = [[] for _ in range(nr_parties)]
        for share in x_reshared:
            for party, share_ptr in enumerate(share.share_ptrs):
                args[party].append(share_ptr)

        shares_p0 = [decompose(args[0][0].get().tensor, 2**64), decompose(args[0][1].get().tensor, 2**64)]
        shares_p1 = [decompose(args[1][0].get().tensor, 2**64), decompose(args[1][1].get().tensor, 2**64)]
        import pdb; pdb.set_trace()

        """

        #p = parallel_execution(ABY3.sum_shares, session.parties)(args)
        res = MPCTensor(shares=p, session=session, mpc_type="binary", shape=x.shape)
        print("Decode")
        print(res.reconstruct(decode=False))
        print("No de code")
        print(res.reconstruct())
        """

    def circuit_add(x, y):
        pass



    def sum_shares(*shares: List[ShareTensor]) -> List[ShareTensor]:
        """
        Sum the shares such that each party would have in the end a Binary Shared Value

        Args:
            shares (List[ShareTensor)): The shares that each party should add together

        Returns:
            shares: The Binary Shares
        """
        res = ShareTensor(session=shares[0].session)
        res.tensor = share[0].tensor

        for share in shares[1:]:
            res = res + share

        return res
