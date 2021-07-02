"""Falcon Protocol.

Falcon : Honest-Majority Maliciously Secure Framework for Private Deep Learning.
arXiv:2004.02229 [cs.CR]
"""
# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
import torch

from sympc.protocol import ABY3
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
        if self.security_type != other.security_type:
            return False

        if type(self) != type(other):
            return False

        return True

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
            args = list(zip(x.share_ptrs, y.share_ptrs))

            z_shares_ptrs = parallel_execution(
                Falcon.compute_zvalue_and_add_mask, session.parties
            )(args)

            result = MPCTensor(shares=z_shares_ptrs, session=x.session)
            result.shape = MPCTensor._get_shape("mul", x.shape, y.shape)  # for prrs
            result = ABY3.truncate(result, session)

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
