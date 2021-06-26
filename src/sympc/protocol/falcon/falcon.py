"""Falcon Protocol.

Falcon : Honest-Majority Maliciously Secure Framework for Private Deep Learning.
arXiv:2004.02229 [cs.CR]
"""
# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# third party
import torch
import torchcsprng as csprng

from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.session import get_session
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor.tensor import SyMPCTensor
from sympc.utils import parallel_execution


class Falcon(metaclass=Protocol):
    """Falcon Protocol Implementation."""

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

    @staticmethod
    def distribute_shares(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        """Forward the call to the tensor specific class.

        Args:
            *args (List[Any]): list of args to be forwarded
            **kwargs(Dict[str, Any): list of named args to be forwarded

        Returns:
            The result returned by the tensor specific distribute_shares method
        """
        return Falcon.share_class.distribute_shares(*args, **kwargs)

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of attributes to compare.

        Args:
            other (Any): Object to compare

        Returns:
            bool: True if equal False if not.
        """
        if not self.security_type == other.security_type:
            return False

        if not type(self).__name__ == type(other).__name__:
            return False

        return True

    @staticmethod
    def trunc1(
        ptr_list: List["ReplicatedSharedTensor"], shape: torch.Size, session: Session
    ) -> List["ReplicatedSharedTensor"]:
        """Apply the trunc1 algorithm in ABY3 for preprocessing.

        Args:
            ptr_list (List[ReplicatedSharedTensor]): Tensor to truncate
            shape(torch.Size) : shape of tensor values
            session(Session) : session the tensor belong to

        Returns:
            List["ReplicatedSharedTensor"] : Truncated shares.
        """
        gen = csprng.create_random_device_generator()
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
    def getTruncationPair(x: MPCTensor, session: Session) -> Tuple[MPCTensor]:
        """Generates truncation pair for the given MPCTensor.

        Args:
            x (MPCTensor): input tensor
            session (Session) : session of the input tensor.

        Returns:
            Tuple[MPCTensor]: generate truncation pair.
        """
        r: List = []
        rPrime: List = []
        for session_ptrs in session.session_ptrs:
            rst = session_ptrs.prrs_generate_random_share(shape=x.shape)
            rst = rst.resolve_pointer_type()
            rPrime.append(rst)
            r.append(rst)

        r = Falcon.trunc1(r, x.shape, session)
        r_mpc = MPCTensor(shares=r, session=session, shape=x.shape)
        rPrime_mpc = MPCTensor(shares=rPrime, session=session, shape=x.shape)
        return r_mpc, rPrime_mpc

    @staticmethod
    def truncate(x: MPCTensor, session: Session) -> MPCTensor:
        """Truncates the MPCTensor by scale factor.

        Args:
            x (MPCTensor): input tensor
            session (Session) : session of the input tensor.

        Returns:
            MPCTensor: truncated MPCTensor.
        """
        x.share_ptrs = Falcon.trunc1(x.share_ptrs, x.shape, session)

        """r,rPrime = Falcon.getTruncationPair(x,session)
        shares: List = []
        scale =(session.config.encoder_base**session.config.encoder_precision)
        op = getattr(operator,"sub")
        x_rp = op(x,rPrime)
        x_rp = x_rp.reconstruct()//scale
        result = r+ x_rp"""

        return x

    @staticmethod
    def mul_master(x: MPCTensor, y: MPCTensor, session: Session) -> MPCTensor:
        """Master method for multiplication.

        Performs Falcon's mul implementation, gets and reshares mul results and distributes shares.
        This needs to be improved in future, it relies on orchestrator being a trusted third party.
        Falcon, requires parties to be able to communication between each other.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to

        Returns:
            shares (ReplicatedSharedTensor): Shares in terms of ReplicatedSharedTensor.

        Raises:
            ValueError: Raised when number of parties are not three.
            NotImplementedError: Raised when implementation not present

        """
        if len(session.parties) != 3:
            raise ValueError("Falcon requires 3 parties")

        result = None

        if session.protocol.security_type == "semi-honest":
            args = []
            for index in range(0, 3):
                args.append(
                    [
                        x.share_ptrs[index],
                        y.share_ptrs[index],
                    ]
                )
            z_shares_ptrs = parallel_execution(
                Falcon.compute_zvalue_and_add_mask, session.parties
            )(args)

            z_shares = [share.get() for share in z_shares_ptrs]

            # Convert 3-3 shares to 2-3 shares by resharing
            reshared_shares = ReplicatedSharedTensor.distribute_shares(
                z_shares, x.session
            )
            result = MPCTensor(shares=reshared_shares, session=x.session)
            result.shape = MPCTensor._get_shape("mul", x.shape, y.shape)  # for prrs
            result = Falcon.truncate(result, session)

        else:
            raise NotImplementedError(
                f"mult operation not implemented for {session.protocol.security_type} setting"
            )

        return result

    @staticmethod
    def compute_zvalue_and_add_mask(
        x: ReplicatedSharedTensor,
        y: ReplicatedSharedTensor,
    ) -> torch.Tensor:
        """Operation to compute local z share and add mask to it.

        Args:
            x (ReplicatedSharedTensor): Secret.
            y (ReplicatedSharedTensor): Another secret.

        Returns:
            share (Torch.tensor): The masked local z share.
        """
        # Parties calculate z value locally
        session = get_session(x.session_uuid)
        z_value = x * y
        przs_mask = session.przs_generate_random_share(shape=x.shape)
        # Add PRZS Mask to z  value
        share = z_value.get_shares()[0] + przs_mask.get_shares()[0]
        return share

    @staticmethod
    def multiplication_protocol(
        x: ReplicatedSharedTensor, y: ReplicatedSharedTensor
    ) -> ReplicatedSharedTensor:
        """Implementation of Falcon's multiplication with semi-honest security guarantee.

        Args:
            x (ReplicatedSharedTensor): Secret
            y (ReplicatedSharedTensor): Another secret

        Returns:
            shares (ReplicatedSharedTensor): results in terms of ReplicatedSharedTensor.
        """
        z_value = (
            x.shares[0] * y.shares[0]
            + x.shares[1] * y.shares[0]
            + x.shares[0] * y.shares[1]
        )
        return z_value
