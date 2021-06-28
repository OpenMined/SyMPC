"""ABY3 Protocol.

ABY3 : A Mixed Protocol Framework for Machine Learning.
https://eprint.iacr.org/2018/403.pdf
"""
# stdlib
from typing import Any
from typing import List
from typing import Tuple

# third party
import torch
import torchcsprng as csprng

from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor.tensor import SyMPCTensor

gen = csprng.create_random_device_generator()


class ABY3(metaclass=Protocol):
    """ABY3 Protocol Implementation."""

    """Used for Share Level static operations like distributing the shares."""
    share_class: SyMPCTensor = ReplicatedSharedTensor
    security_levels: List[str] = ["semi-honest", "malicious"]

    def __init__(self, security_type: str = "semi-honest"):
        """Initialization of the Protocol.

        Args:
            security_type : specifies the security level of the Protocol.

        Raises:
            ValueError : If invalid security_type is provided.
        """
        if security_type not in self.security_levels:
            raise ValueError(f"{security_type} is not a valid security type")

        self.security_type = security_type

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of attributes to compare.

        Args:
            other (Any): Object to compare

        Returns:
            bool: True if equal False if not.
        """
        if self.security_type != other.security_type:
            return False

        if type(self) != type(other):
            return False

        return True

    @staticmethod
    def trunc1(
        ptr_list: List["ReplicatedSharedTensor"], shape: torch.Size, session: Session
    ) -> List["ReplicatedSharedTensor"]:
        """Performs the ABY3 trunc1 algorithm.

        Args:
            ptr_list (List[ReplicatedSharedTensor]): Tensors to truncate
            shape(torch.Size) : shape of tensor values
            session(Session) : session the tensor belong to

        Returns:
            List["ReplicatedSharedTensor"] : Truncated shares.
        """
        rand_value = torch.empty(size=shape, dtype=session.tensor_type).random_(
            generator=gen
        )
        base = session.config.encoder_base
        precision = session.config.encoder_precision
        scale = base ** precision
        x1 = ptr_list[0].get_copy().shares[0]
        x2, x3 = ptr_list[1].get_copy().shares
        x1_trunc = x1 >> precision if base == 2 else x1 // scale
        x_trunc = (x2 + x3) >> precision if base == 2 else (x2 + x3) // scale
        shares = [x1_trunc, x_trunc - rand_value, rand_value]
        ptr_list = ReplicatedSharedTensor.distribute_shares(shares, session)
        return ptr_list

    @staticmethod
    def truncate(x: MPCTensor, session: Session) -> MPCTensor:
        """Perfoms the ABY3 truncation algorithm.

        Args:
            x (MPCTensor): input tensor
            session (Session) : session of the input tensor.

        Returns:
            MPCTensor: truncated MPCTensor.

        Raises:
            ValueError : parties involved in the computation is not equal to three.

        TODO :Switch to trunc2 algorithm  as it is communication efficient.
        """
        if session.nr_parties != 3:
            raise ValueError("Share trunc1 algorithm works only for 3 parites.")

        share_ptrs = ABY3.trunc1(x.share_ptrs, x.shape, session)
        result = MPCTensor(shares=share_ptrs, session=session, shape=x.shape)

        return result

    @staticmethod
    def trunc2(x: MPCTensor, session: Session) -> MPCTensor:
        """Truncates the MPCTensor by scale factor using trunc2 algorithm.

        Args:
            x (MPCTensor): input tensor
            session (Session) : session of the input tensor.

        Returns:
            MPCTensor: truncated MPCTensor.

        TODO : The trunc2 algorithm is erroneous, to be optimized.
        """
        r, rPrime = ABY3.get_truncation_pair(x, session)
        scale = session.config.encoder_base ** session.config.encoder_precision
        # op = getattr(operator,"sub")
        x_rp = x - rPrime
        x_rp = x_rp.reconstruct(decode=False) // scale
        zero = torch.tensor([0])
        x_rp = MPCTensor(shares=[x_rp, zero, zero], session=x.session, shape=x.shape)

        result = r + x_rp

        return result

    @staticmethod
    def get_truncation_pair(x: MPCTensor, session: Session) -> Tuple[MPCTensor]:
        """Generates truncation pair for the given MPCTensor.

        Args:
            x (MPCTensor): input tensor
            session (Session) : session of the input tensor.

        Returns:
            Tuple[MPCTensor]: generated truncation pair.
        """
        r: List = []
        rPrime: List = []
        for session_ptrs in session.session_ptrs:
            rst = session_ptrs.prrs_generate_random_share(shape=x.shape)
            rst = rst.resolve_pointer_type()
            rPrime.append(rst)
            r.append(rst)

        r = ABY3.trunc1(r, x.shape, session)
        r_mpc = MPCTensor(shares=r, session=session, shape=x.shape)
        rPrime_mpc = MPCTensor(shares=rPrime, session=session, shape=x.shape)
        return r_mpc, rPrime_mpc
