from typing import List

from sympc.protocol import Protocol
from sympc.tensor import MPCTensor
from sympc.utils import decompose

from copy import deepcopy


class ABY3(metaclass=Protocol):

    @staticmethod
    def A2B(x: MPCTensor) -> List[MPCTensor]:
        """
        Convert an Arithmetic Shared Value to a Binary Shared Value
        (from ring size 2^(pow != 1) to ring size 2)
        """

        session = x.session
        ring_size = session.ring_size

        import pdb; pdb.set_trace()
        x_decomposed_shares = [
            decompose(share, shape=x.shape, ring_size=session.ring_size) for share in x.share_ptrs]
        session.ring_size = 2
        x_share_binary = [
                MPCTensor(session=session, secret=x_share)]
        MPCTensor.generate_przs(x.shape, session)
