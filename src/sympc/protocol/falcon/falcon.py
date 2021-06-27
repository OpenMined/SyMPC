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

from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.session import get_session
from sympc.store import CryptoPrimitiveProvider
from sympc.store.exceptions import EmptyPrimitiveStore
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

        if not type(self) == type(other):
            return False

        return True

    @staticmethod
    def mul_master(x: MPCTensor, y: MPCTensor, session: Session) -> MPCTensor:
        """Master method for multiplication.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to

        Returns:
            MPCTensor: Result of the operation.

        Raises:
            ValueError: Raised when number of parties are not three.
            ValueError : Raised when invalid security_type is provided.
        """
        from sympc.protocol import ABY3

        if len(session.parties) != 3:
            raise ValueError("Falcon requires 3 parties")

        result = None

        if session.protocol.security_type == "semi-honest":
            result = Falcon.mul_semi_honest(x, y, session)
        elif session.protocol.security_type == "malicious":
            result = Falcon.mul_malicious(x, y, session)
        else:
            raise ValueError("Invalid security_type for Falcon multiplication")

        result = ABY3.truncate(result, session)
        return result

    @staticmethod
    def mul_semi_honest(x: MPCTensor, y: MPCTensor, session: Session) -> MPCTensor:
        """Falcon semihonest multiplication.

        Performs Falcon's mul implementation, gets and reshares mul results and distributes shares.
        This needs to be improved in future, it relies on orchestrator being a trusted third party.
        Falcon, requires parties to be able to communication between each other.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to

        Returns:
            MPCTensor: Result of the operation.
        """
        result = None
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
        reshared_shares = ReplicatedSharedTensor.distribute_shares(z_shares, x.session)
        result = MPCTensor(shares=reshared_shares, session=x.session)
        result.shape = MPCTensor._get_shape("mul", x.shape, y.shape)  # for prrs

        return result

    @staticmethod
    def triple_verification(
        z_sh: ReplicatedSharedTensor, eps: torch.Tensor, delta: torch.Tensor
    ) -> ReplicatedSharedTensor:
        """Performs Beaver's triple verification check.

        Args:
            z_sh (ReplicatedSharedTensor) : share of multiplied value(x*y).
            eps (torch.Tensor) :masked value of x
            delta (torch.Tensor): masked value of y

        Returns:
            ReplicatedSharedTensor : Result of the verification.
        """
        session = get_session(z_sh.session_uuid)

        crypto_store = session.crypto_store
        eps_shape = tuple(eps.shape)
        delta_shape = tuple(delta.shape)

        op_str = "mul"
        primitives = crypto_store.get_primitives_from_store(
            f"beaver_{op_str}", eps_shape, delta_shape
        )

        a_share, b_share, c_share = primitives

        eps_b = b_share * eps
        delta_a = a_share * delta
        eps_delta = eps * delta

        rst_share = z_sh - c_share - delta_a - eps_b
        rst_share = rst_share + eps_delta

        return rst_share

    @staticmethod
    def falcon_mask(
        x_sh: ReplicatedSharedTensor, y_sh: ReplicatedSharedTensor
    ) -> Tuple[ReplicatedSharedTensor, ReplicatedSharedTensor]:
        """Falcon mask.

        Args:
            x_sh (ReplicatedSharedTensor): X share
            y_sh (ReplicatedSharedTensor) : Y share

        Returns:
            Tuple[ReplicatedSharedTensor,ReplicatedSharedTensor] : Masked values.
        """
        session = get_session(x_sh.session_uuid)

        crypto_store = session.crypto_store

        op_str = "mul"
        primitives = crypto_store.get_primitives_from_store(
            f"beaver_{op_str}", x_sh.shape, y_sh.shape, remove=False
        )

        a_sh, b_sh, _ = primitives

        return x_sh - a_sh, y_sh - b_sh

    @staticmethod
    def mul_malicious(x: MPCTensor, y: MPCTensor, session: Session) -> MPCTensor:
        """Falcon malicious multiplication.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to

        Returns:
            MPCTensor: Result of the operation.

        Raises:
            ValueError : If the shares are not valid.
        """
        shape_x = tuple(x.shape)
        shape_y = tuple(y.shape)

        result = Falcon.mul_semi_honest(x, y, session)
        print(result.reconstruct())
        args = [list(sh) for sh in zip(x.share_ptrs, y.share_ptrs)]
        op_str = "mul"
        try:
            mask = parallel_execution(Falcon.falcon_mask, session.parties)(args)
        except EmptyPrimitiveStore:
            CryptoPrimitiveProvider.generate_primitives(
                f"beaver_{op_str}",
                session=session,
                g_kwargs={
                    "session": session,
                    "a_shape": shape_x,
                    "b_shape": shape_y,
                    "nr_parties": session.nr_parties,
                    # **kwargs_,
                },
                p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
            )
            mask = parallel_execution(Falcon.falcon_mask, session.parties)(args)

        eps_shares, delta_shares = zip(*mask)

        eps = MPCTensor(shares=eps_shares, session=session)
        delta = MPCTensor(shares=delta_shares, session=session)

        eps_plaintext = eps.reconstruct(decode=False)
        delta_plaintext = delta.reconstruct(decode=False)

        args = [
            list(sh) + [eps_plaintext, delta_plaintext] for sh in zip(result.share_ptrs)
        ]

        triple_shares = parallel_execution(Falcon.triple_verification, session.parties)(
            args
        )

        triple = MPCTensor(shares=triple_shares, session=x.session)

        if triple.reconstruct(decode=False) == 0:
            return result
        else:
            raise ValueError("Computation Aborted: Malicious behavior.")

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
