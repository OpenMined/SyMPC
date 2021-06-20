"""Falcon Protocol.

Falcon : Honest-Majority Maliciously Secure Framework for Private Deep Learning.
arXiv:2004.02229 [cs.CR]
"""
# stdlib
from typing import Any
from typing import Dict
from typing import List

from sympc.protocol.protocol import Protocol
from sympc.session import Session
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

            def multiply_shares(party_rank):
                return x.share_ptrs[party_rank] * y.share_ptrs[party_rank]

            z_shares = parallel_execution(multiply_shares)([[0], [1], [2]])

            reshared_shares = []

            def get_shares_and_add_mask(party_rank):
                przs_mask = (
                    session.session_ptrs[party_rank]
                    .przs_generate_random_share(shape=x.shape)
                    .get_shares()
                    .get()[0]
                )
                share = z_shares[party_rank].get_shares().get()[0] + przs_mask
                return share

            shares = parallel_execution(get_shares_and_add_mask)([[0], [1], [2]])
            reshared_shares = ReplicatedSharedTensor.distribute_shares(
                shares, x.session
            )
            result = MPCTensor(shares=reshared_shares, session=x.session)

        else:
            raise NotImplementedError(
                f"mult operation not implemented for {session.protocol.security_type} setting"
            )

        return result

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
