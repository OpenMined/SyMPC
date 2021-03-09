# third party
import torch
import torchcsprng as csprng

import sympc
from sympc.protocol.protocol import Protocol
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import decompose
from sympc.utils import parallel_execution

generator = csprng.create_random_device_generator()


class SecureNN(metaclass=Protocol):
    @staticmethod
    def private_compare(x_bit: MPCTensor, r: torch.Tensor, l: int) -> MPCTensor:
        """Perform privately x > r.

        Args:
            x (MPCTensor): the private tensor
            r (torch.Tensor): the value to compare with
            beta (torch.Tensor): a boolean commonly held by the parties to
                hide the result of computation
            l (int): field size for r

        Return:
            beta' = beta xor (x > r).
        """

        session = x_bit.session
        parties = session.parties
        tensor_type = session.tensor_type
        beta = torch.empty(size=x_bit.shape, dtype=torch.bool).random_(
            generator=generator
        )

        # the commented out numbers below correspond to the
        # line numbers in Algorithm 3 of the SecureNN paper
        # https://eprint.iacr.org/2018/442.pdf

        # Common randomess
        s = torch.empty(size=x_bit.shape, dtype=tensor_type).random_(
            generator=generator
        )
        u = torch.empty(size=x_bit.shape, dtype=tensor_type).random_(
            generator=generator
        )

        perm = torch.randperm(x_bit.shape[-1])

        # 1)
        t = r + 1

        t_bit = decompose(t, l)
        r_bit = decompose(r, l)

        args = [list(el) for el in zip(session.session_ptrs, x_bit.share_ptrs)]
        common_args = [beta, s, u, perm, t_bit, r_bit]

        for arg in args:
            arg.extend(common_args)

        shares = parallel_execution(SecureNN.private_compare_parties, session.parties)(
            args
        )

        return None

    @staticmethod
    def private_compare_parties(
        session, x_bit_share, beta, s, u, perm, t_bit, r_bit
    ) -> ShareTensor:
        if session.rank == 0:
            print(r_bit)
            print(x_bit_share)
        w = x_bit_share + (session.rank * r_bit) - (2 * r_bit * x_bit_share)

        wc = w.flip(-1).cumsum(-1).flip(-1) - w
        c_beta0 = -x_bit_share + (session.rank * r_bit) + session.rank + wc

        return None

    @staticmethod
    def share_convert_parties(share: ShareTensor) -> ShareTensor:
        session = x.session
        ring_size = session.riing_size

        przs_share = session.przs_generate_random_share(share, share.tensor.shape)
        res = przs_share + share
        return res

    @staticmethod
    def relu_deriv(x: MPCTensor) -> MPCTensor:
        """Compute the derivative of ReLu.

        Args:
            x (MPCTensor): the privately shared tensor

        Return:
            0 if Dec(x) < 0
            1 if Dec(x) > 0
        """
        session = x.session
        parties = session.parties
        ring_size = session.ring_size

        pass
