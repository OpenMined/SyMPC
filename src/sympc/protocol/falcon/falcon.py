"""Falcon Protocol.

Falcon : Honest-Majority Maliciously Secure Framework for Private Deep Learning.
arXiv:2004.02229 [cs.CR]
"""
# stdlib
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

# third party
import torch

from sympc.config import Config
from sympc.protocol import ABY3
from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.session import get_session
from sympc.store import CryptoPrimitiveProvider
from sympc.store.exceptions import EmptyPrimitiveStore
from sympc.tensor import MPCTensor
from sympc.tensor import PRIME_NUMBER
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor.tensor import SyMPCTensor
from sympc.utils import get_type_from_ring
from sympc.utils import parallel_execution

shares_sum = ReplicatedSharedTensor.shares_sum


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
    def mul_master(
        x: MPCTensor,
        y: MPCTensor,
        session: Session,
        op_str: str,
        kwargs_: Dict[Any, Any],
    ) -> MPCTensor:
        """Master method for multiplication.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to
            op_str (str): Operation string.
            kwargs_ (Dict[Any, Any]): Kwargs for some operations like conv2d

        Returns:
            result (MPCTensor): Result of the operation.

        Raises:
            ValueError: Raised when number of parties are not three.
            ValueError : Raised when invalid security_type is provided.
        """
        if len(session.parties) != 3:
            raise ValueError("Falcon requires 3 parties")

        result = None

        ring_size = int(x.share_ptrs[0].get_ring_size().get_copy())
        conf_dict = x.share_ptrs[0].get_config().get_copy()
        config = Config(**conf_dict)

        if session.protocol.security_type == "semi-honest":
            result = Falcon.mul_semi_honest(
                x, y, session, op_str, ring_size, config, **kwargs_
            )
        elif session.protocol.security_type == "malicious":
            result = Falcon.mul_malicious(
                x, y, session, op_str, ring_size, config, **kwargs_
            )
        else:
            raise ValueError("Invalid security_type for Falcon multiplication")

        result = ABY3.truncate(result, session, ring_size, config)

        return result

    @staticmethod
    def mul_semi_honest(
        x: MPCTensor,
        y: MPCTensor,
        session: Session,
        op_str: str,
        ring_size: int,
        config: Config,
        reshare: bool = False,
        **kwargs_: Dict[Any, Any],
    ) -> MPCTensor:
        """Falcon semihonest multiplication.

        Performs Falcon's mul implementation, add masks and performs resharing.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to
            op_str (str): Operation string.
            ring_size (int) : Ring size of the underlying tensors.
            config (Config): The configuration(base,precision) of the underlying tensor.
            reshare (bool) : Convert 3-out-3 to 2-out-3 if set.
            kwargs_ (Dict[Any, Any]): Kwargs for some operations like conv2d

        Returns:
            MPCTensor: Result of the operation.
        """
        args = [
            [x_share, y_share, op_str]
            for x_share, y_share in zip(x.share_ptrs, y.share_ptrs)
        ]

        z_shares_ptrs = parallel_execution(
            Falcon.compute_zvalue_and_add_mask, session.parties
        )(args, kwargs_)

        result = MPCTensor(shares=z_shares_ptrs, session=x.session)

        if reshare:
            z_shares = [share.get() for share in z_shares_ptrs]

            # Convert 3-3 shares to 2-3 shares by resharing
            reshared_shares = ReplicatedSharedTensor.distribute_shares(
                z_shares, x.session, ring_size, config
            )
            result = MPCTensor(shares=reshared_shares, session=x.session)

        result.shape = MPCTensor._get_shape(op_str, x.shape, y.shape)  # for prrs
        return result

    @staticmethod
    def triple_verification(
        z_sh: ReplicatedSharedTensor,
        eps: torch.Tensor,
        delta: torch.Tensor,
        op_str: str,
        **kwargs: Dict[Any, Any],
    ) -> ReplicatedSharedTensor:
        """Performs Beaver's triple verification check.

        Args:
            z_sh (ReplicatedSharedTensor) : share of multiplied value(x*y).
            eps (torch.Tensor) :masked value of x
            delta (torch.Tensor): masked value of y
            op_str (str): Operator string.
            kwargs (Dict[Any, Any]): Keywords arguments for the operator.

        Returns:
            ReplicatedSharedTensor : Result of the verification.
        """
        session = get_session(z_sh.session_uuid)
        ring_size = z_sh.ring_size

        crypto_store = session.crypto_store
        eps_shape = tuple(eps.shape)
        delta_shape = tuple(delta.shape)

        primitives = crypto_store.get_primitives_from_store(
            f"beaver_{op_str}", eps_shape, delta_shape
        )

        a_share, b_share, c_share = primitives

        op = ReplicatedSharedTensor.get_op(ring_size, op_str)

        eps_delta = op(eps, delta, **kwargs)
        eps_b = b_share.clone()
        delta_a = a_share.clone()

        # prevent re-encoding as the values are encoded.
        # TODO: should be improved.
        for i in range(2):
            eps_b.shares[i] = op(eps, eps_b.shares[i])
            delta_a.shares[i] = op(delta_a.shares[i], delta)

        rst_share = c_share + delta_a + eps_b

        if session.rank == 0:
            rst_share.shares[0] = shares_sum(
                [rst_share.shares[0], eps_delta], ring_size
            )

        if session.rank == 2:
            rst_share.shares[1] = shares_sum(
                [rst_share.shares[1], eps_delta], ring_size
            )

        result = z_sh - rst_share

        return result

    @staticmethod
    def falcon_mask(
        x_sh: ReplicatedSharedTensor, y_sh: ReplicatedSharedTensor, op_str: str
    ) -> Tuple[ReplicatedSharedTensor, ReplicatedSharedTensor]:
        """Falcon mask.

        Args:
            x_sh (ReplicatedSharedTensor): X share
            y_sh (ReplicatedSharedTensor) : Y share
            op_str (str): Operator

        Returns:
            values(Tuple[ReplicatedSharedTensor,ReplicatedSharedTensor]) : masked_values.
        """
        session = get_session(x_sh.session_uuid)

        crypto_store = session.crypto_store

        primitives = crypto_store.get_primitives_from_store(
            f"beaver_{op_str}", x_sh.shape, y_sh.shape, remove=False
        )

        a_sh, b_sh, _ = primitives

        return x_sh - a_sh, y_sh - b_sh

    @staticmethod
    def mul_malicious(
        x: MPCTensor,
        y: MPCTensor,
        session: Session,
        op_str: str,
        ring_size: int,
        config: Config,
        **kwargs_: Dict[Any, Any],
    ) -> MPCTensor:
        """Falcon malicious multiplication.

        Args:
            x (MPCTensor): Secret
            y (MPCTensor): Another secret
            session (Session): Session the tensors belong to
            op_str (str): Operation string.
            ring_size (int) : Ring size of the underlying tensor.
            config (Config): The configuration(base,precision) of the underlying tensor.
            kwargs_ (Dict[Any, Any]): Kwargs for some operations like conv2d

        Returns:
            result(MPCTensor): Result of the operation.

        Raises:
            ValueError : If the shares are not valid.
        """
        shape_x = tuple(x.shape)
        shape_y = tuple(y.shape)

        result = Falcon.mul_semi_honest(
            x, y, session, op_str, ring_size, config, reshare=True, **kwargs_
        )

        args = [list(sh) + [op_str] for sh in zip(x.share_ptrs, y.share_ptrs)]
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
                    "ring_size": ring_size,
                    "config": config,
                    **kwargs_,
                },
                p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
            )
            mask = parallel_execution(Falcon.falcon_mask, session.parties)(args)

        # zip on pointers is compute intensive
        mask_local = [mask[idx].get() for idx in range(session.nr_parties)]
        eps_shares, delta_shares = zip(*mask_local)

        eps_plaintext = ReplicatedSharedTensor.reconstruct(eps_shares)
        delta_plaintext = ReplicatedSharedTensor.reconstruct(delta_shares)

        args = [
            list(sh) + [eps_plaintext, delta_plaintext, op_str]
            for sh in zip(result.share_ptrs)
        ]

        triple_shares = parallel_execution(Falcon.triple_verification, session.parties)(
            args, kwargs_
        )

        triple = MPCTensor(shares=triple_shares, session=x.session)

        if (triple.reconstruct(decode=False) == 0).all():
            return result
        else:
            raise ValueError("Computation Aborted: Malicious behavior.")

    @staticmethod
    def compute_zvalue_and_add_mask(
        x: ReplicatedSharedTensor,
        y: ReplicatedSharedTensor,
        op_str: str,
        **kwargs: Dict[Any, Any],
    ) -> torch.Tensor:
        """Operation to compute local z share and add mask to it.

        Args:
            x (ReplicatedSharedTensor): Secret.
            y (ReplicatedSharedTensor): Another secret.
            op_str (str): Operation string.
            kwargs (Dict[Any, Any]): Kwargs for some operations like conv2d

        Returns:
            share (Torch.tensor): The masked local z share.
        """
        # Parties calculate z value locally
        session = get_session(x.session_uuid)
        z_value = Falcon.multiplication_protocol(x, y, op_str, **kwargs)
        shape = MPCTensor._get_shape(op_str, x.shape, y.shape)
        przs_mask = session.przs_generate_random_share(
            shape=shape, ring_size=str(x.ring_size)
        )
        # Add PRZS Mask to z  value
        op = ReplicatedSharedTensor.get_op(x.ring_size, "add")
        share = op(z_value, przs_mask.get_shares()[0])

        return share

    @staticmethod
    def multiplication_protocol(
        x: ReplicatedSharedTensor,
        y: ReplicatedSharedTensor,
        op_str: str,
        **kwargs: Dict[Any, Any],
    ) -> ReplicatedSharedTensor:
        """Implementation of Falcon's multiplication with semi-honest security guarantee.

        Args:
            x (ReplicatedSharedTensor): Secret
            y (ReplicatedSharedTensor): Another secret
            op_str (str): Operator string.
            kwargs (Dict[Any, Any]): Keywords arguments for the operator.

        Returns:
            shares (ReplicatedSharedTensor): results in terms of ReplicatedSharedTensor.
        """
        op = ReplicatedSharedTensor.get_op(x.ring_size, op_str)

        z_value = shares_sum(
            [
                op(x.shares[0], y.shares[0], **kwargs),
                op(x.shares[1], y.shares[0], **kwargs),
                op(x.shares[0], y.shares[1], **kwargs),
            ],
            x.ring_size,
        )
        return z_value

    @staticmethod
    def select_shares(x: MPCTensor, y: MPCTensor, b: MPCTensor) -> MPCTensor:
        """Returns either x or y based on bit b.

        Args:
            x (MPCTensor): input tensor
            y (MPCTensor): input tensor
            b (MPCTensor): input tensor which is shares of a bit used as selector bit.

        Returns:
            z (MPCTensor):Returns x (if b==0) or y (if b==1).

        Raises:
            ValueError: If the selector bit tensor is not of ring size "2".
        """
        ring_size = int(b.share_ptrs[0].get_ring_size().get_copy())
        shape = b.shape
        if ring_size != 2:
            raise ValueError(
                f"Invalid {ring_size} for selector bit,must be of ring size 2"
            )
        if shape is None:
            raise ValueError("The selector bit tensor must have a valid shape.")
        session = x.session

        # TODO: Should be made to generate with CryptoProvider in Preprocessing stage.
        c_ptrs: List[ReplicatedSharedTensor] = []
        for session_ptr in session.session_ptrs:
            c_ptrs.append(
                session_ptr.prrs_generate_random_share(
                    shape=shape, ring_size=str(ring_size)
                )
            )

        c = MPCTensor(shares=c_ptrs, session=session, shape=shape)  # bit random share
        c_r = ABY3.bit_injection(
            c, session, session.ring_size
        )  # bit random share in session ring.

        tensor_type = get_type_from_ring(session.ring_size)
        mask = (b ^ c).reconstruct(decode=False).type(tensor_type)

        d = (mask - (c_r * mask)) + (c_r * (mask ^ 1))

        # Order placed carefully to prevent re-encoding,should not be changed.
        z = x + (d * (y - x))

        return z

    @staticmethod
    def _random_prime_group(
        session: Session, shape: Union[torch.Size, tuple]
    ) -> MPCTensor:
        """Computes shares of random number in Zp*.Zp* is the multiplicative group mod p.

        Args:
            session (Session): session to generate random shares for.
            shape (Union[torch.Size, tuple]): shape of the random share to generate.

        Returns:
            share (MPCTensor): Returns shares of random number in group Zp*.

        Zp* = {1,2..,p-1},where p is a prime number.
        We use Euler's Theorem for verifying that random share is not zero.
        It states that:
        For a general modulus n
        a^phi(n) = 1(mod n), if a is co prime to n.
        In our case n=p(prime number), phi(p) = p-1
        phi(n) = Euler totient function.
        We effectively try to sample a random number in range [1,p-1],discard the instances where
        it equals zero.
        """
        while True:
            ptr_list: List[ReplicatedSharedTensor] = []
            for session_ptr in session.session_ptrs:
                ptr = session_ptr.prrs_generate_random_share(
                    shape=(), ring_size=str(PRIME_NUMBER)
                ).resolve_pointer_type()
                ptr = ptr.repeat(shape)
                ptr_list.append(ptr)

            m = MPCTensor(shares=ptr_list, session=session, shape=shape)

            m_euler = m ** (PRIME_NUMBER - 1)

            if (m_euler.reconstruct(decode=False) == 1).all():
                return m

    @staticmethod
    def private_compare(x: List[MPCTensor], r: torch.Tensor) -> MPCTensor:
        """Falcon Private Compare functionality which computes(x>r).

        Args:
            x (List[MPCTensor]) : shares of bits of x in Zp.
            r (torch.Tensor) : Public value r.

        Returns:
            result (MPCTensor): Returns shares of bits of the operation.

        Raises:
            ValueError: If input shares is not a list.
            ValueError: If input public value is not a tensor.

        (if (x>=r) returns 1 else returns 0)
        """
        if not isinstance(x, list):
            raise ValueError(f"Input shares for Private Compare: {x} must be a list")

        if not isinstance(r, torch.Tensor):
            raise ValueError(f"Value r:{r} must be a torch tensor for private compare")

        shape = x[0].shape
        session = x[0].session

        ptr_list: List[ReplicatedSharedTensor] = [
            session_ptr.prrs_generate_random_share(shape=shape, ring_size="2")
            for session_ptr in session.session_ptrs
        ]

        beta_2 = MPCTensor(
            shares=ptr_list, session=session, shape=shape
        )  # shares of random bit
        beta_p = ABY3.bit_injection(
            beta_2, session, PRIME_NUMBER
        )  # shares of random bit in Zp.
        m = Falcon._random_prime_group(session, shape)

        nr_shares = len(x)
        u = [0] * nr_shares
        c = [0] * nr_shares

        w = 0

        for i in range(len(x) - 1, -1, -1):
            r_i = (r >> i) & 1  # bit at ith position
            u[i] = (1 - 2 * beta_p) * (x[i] - r_i)
            c[i] = u[i] + 1 + w
            w += x[i] ^ r_i

        d = m * math.prod(c)

        d_val = d.reconstruct(decode=False)  # plaintext d.
        d_val[d_val != 0] = 1  # making all non zero values as 1.

        beta_prime = d_val

        return beta_2 + beta_prime
