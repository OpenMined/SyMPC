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
import numpy as np
import torch
import torchcsprng as csprng

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
from sympc.utils import get_nr_bits
from sympc.utils import get_type_from_ring
from sympc.utils import parallel_execution

shares_sum = ReplicatedSharedTensor.shares_sum
gen = csprng.create_random_device_generator()
UNSIGNED_MAP = {
    torch.bool: "torch.bool",
    torch.int8: "uint8",
    torch.int16: "uint16",
    torch.int32: "uint32",
    torch.int64: "uint64",
}


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

        if isinstance(z_sh.shares[0], np.ndarray):
            dtype = str(z_sh.shares[0].dtype)
            eps_b, delta_a, c_share = (
                eps_b.to_numpy(dtype),
                delta_a.to_numpy(dtype),
                c_share.to_numpy(dtype),
            )

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
        eps_shares = [mask[idx][0] for idx in range(session.nr_parties)]
        delta_shares = [mask[idx][1] for idx in range(session.nr_parties)]

        eps = MPCTensor(shares=eps_shares, session=session)
        delta = MPCTensor(shares=delta_shares, session=session)

        eps_plaintext = eps.reconstruct(decode=False)
        delta_plaintext = delta.reconstruct(decode=False)

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
        przs_mask = przs_mask.get_shares()[0]

        if isinstance(z_value, np.ndarray):
            przs_mask = przs_mask.numpy().astype(z_value.dtype)

        share = op(z_value, przs_mask)

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
        """Computes shares of random number in Zp*.Zp* is the multplicative group mod p.Zp* = {1,2..,p-1}.

        Args:
            session (Session): session to generate random shares for.
            shape (Union[torch.Size,tuple]): shape of the random share to generate.

        Returns:
            share (MPCTensor): Retuns shares of random number in group Zp*.

        We use Euler's Theorum for verifying that random share is not zero.
        It states that:
        For a general modulus n
        a^phi(n) = 1(mod n), if a is co prime to n.
        In our case n=p(prime number), phi(p) = p-1
        phi(n) = Euler totient function.
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
            r (torch.Tensor) : Public integer r.

        Returns:
            result (MPCTensor): Returns shares of bits of the operation.

        (if (x>=r) returns 1 else returns 0)
        """
        shape = x[0].shape
        session = x[0].session
        ptr_list: List = []
        for session_ptr in session.session_ptrs:
            sh_ptr = session_ptr.prrs_generate_random_share(
                shape=shape, ring_size=str(2)
            )
            ptr_list.append(sh_ptr)

        beta_2 = MPCTensor(
            shares=ptr_list, session=session, shape=shape
        )  # shares of random bit
        beta_p = ABY3.bit_injection(
            beta_2, session, PRIME_NUMBER
        )  # shares of random bit in Zp.
        m = Falcon._random_prime_group(session, shape)
        if isinstance(r, np.ndarray):
            beta_2.to_numpy("bool_")
            beta_p.to_numpy("uint8")
            m.to_numpy("uint8")

        u = [0 for i in range(len(x))]
        w = [0 for i in range(len(x))]
        c = [0 for i in range(len(x))]
        for i in range(len(x) - 1, -1, -1):
            r_i = r >> i & 1  # bit at ith position
            u[i] = (1 - 2 * beta_p) * (x[i] - r_i)
            w[i] = x[i] ^ r_i
            c[i] = u[i] + 1 + sum(w[i + 1 :])

        d = m * (math.prod(c))

        d_val = d.reconstruct(decode=False)  # plaintext d.
        d_val[d_val != 0] = 1  # making all non zero values as 1.

        beta_prime = d_val

        return beta_2 + beta_prime

    @staticmethod
    def wrap2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes wrap on the input numpy array.

        Args:
            a (np.ndarray): Input numpy array
            b (np.ndarray): Input numpy array.

        Returns:
            result (np.ndarray): Boolean array ,True if there is a carry.

        Raises:
            ValueError: If the input values is not a numpy array.
            ValueError: If signed numpy array are provided as input.

        wrap2 calucaties the carry bit on addition of two values a+b.
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise ValueError("Input value must be a numpy array for wrap2.")

        if a.dtype.kind == "i" or b.dtype.kind == "i":
            raise ValueError("Wrap2 works only for signed numbers.")

        max_val = np.iinfo(a.dtype).max

        return a > max_val - b

    @staticmethod
    def wrap3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Computes wrap on the input numpy array.

        Args:
            a (np.ndarray): Input numpy array
            b (np.ndarray): Input numpy array.
            c (np.ndarray): Input numpy array.

        Returns:
            result (np.ndarray): Modulo reduction of the exact wrap function.

        Raises:
            ValueError: If the input values is not a numpy array.
            ValueError: If signed numpy array are provided as input.

        wrap3 calucaties the carry bit on addition of three values a+b+c(mod 2).
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise ValueError("Input value must be a numpy array for wrap3.")

        if a.dtype.kind == "i" or b.dtype.kind == "i" or c.dtype.kind == "i":
            raise ValueError("Wrap3 works only for signed numbers.")

        return Falcon.wrap2(a, b) ^ Falcon.wrap2(a + b, c)

    @staticmethod
    def wrap_preprocess(a: MPCTensor, session: Session) -> List:
        """Generates Preprocess values for wrap function.

        Args:
            a (MPCTensor): input tensor
            session (Session) : session the tensors belong to

        Returns:
            x,x_p,x_wrap (List) : Returns a random shares in session ring, prime ring, wrap of it.

        Raises:
            ValueError : If the input tensor shape is None.
        """
        shape = a.shape
        if shape is None:
            raise ValueError("Input MPCTensor for Wrap must have valid shape")
        tensor_type = session.tensor_type
        dtype = UNSIGNED_MAP[tensor_type]
        ring_size = session.ring_size

        ptr_list: List = []
        for session_ptr in session.session_ptrs:
            ptr_list.append(
                session_ptr.prrs_generate_random_share(
                    shape=shape, ring_size=str(ring_size)
                )
            )

        x = MPCTensor(shares=ptr_list, session=session, shape=shape)
        x_b = ABY3.bit_decomposition_ttp(x, session)  # bit sharing
        x_p: List = []  # bit sharing in Zp

        for idx in range(len(x_b)):
            p_sh = ABY3.bit_injection(x_b[idx], session, PRIME_NUMBER)
            p_sh.to_numpy("uint8")
            x_p.append(p_sh)

        x.to_numpy(dtype)
        x1 = x.share_ptrs[0].get_copy().shares[0]
        x2, x3 = x.share_ptrs[1].get_copy().shares

        x_wrap = Falcon.wrap3(x1, x2, x3)

        # numpy random generator generate by shape only for float types.
        r1 = torch.empty(size=shape, dtype=torch.bool).random_(generator=gen).numpy()
        r2 = torch.empty(size=shape, dtype=torch.bool).random_(generator=gen).numpy()
        r3 = x_wrap ^ r1 ^ r2

        x_wrap_sh: List[np.ndarray] = [r1, r2, r3]
        share_ptrs = ReplicatedSharedTensor.distribute_shares(x_wrap_sh, session, 2)

        alpha = MPCTensor(shares=share_ptrs, session=session, shape=shape)

        a.to_numpy(dtype)

        return a, x, x_p, alpha

    @staticmethod
    def wrap(a: MPCTensor) -> MPCTensor:
        """Falcon Wrap functionality which computes wrap on underlying shares.

        Args:
            a (MPCTensor) : input tensor to compute wrap.

        Returns:
            wrap_sh (MPCTensor): bit shares of wrap of the input tensor.
        """
        session = a.session

        a, x, x_p, alpha = Falcon.wrap_preprocess(a, session)

        r = x + a

        # TODO : change to get shares by reconstruct,malicious returns all_shares
        r1 = r.share_ptrs[0].get_copy().shares[0]
        r2, r3 = r.share_ptrs[1].get_copy().shares

        share_ptrs = []
        for a_sh, x_sh in zip(a.share_ptrs, x.share_ptrs):
            share_ptrs.append(a_sh.wrap_rst(x_sh))

        beta = MPCTensor(shares=share_ptrs, session=session, shape=a.shape)

        delta = Falcon.wrap3(r1, r2, r3)

        r_public = r1 + r2 + r3

        eta = Falcon.private_compare(x_p, r_public + 1)

        wrap_sh = beta + delta - eta - alpha

        wrap_sh.from_numpy()

        return wrap_sh

    @staticmethod
    def drelu(a: MPCTensor) -> MPCTensor:
        """Computes Derivative of ReLU on input MPCTensor.

        Args:
            a (MPCTensor): input tensor.

        Returns:
            result (MPCTensor): DReLU of input tensor.

        Raises:
            ValueError: If the input tensor does not have a valid shape.
        """
        session = a.session
        shape = a.shape
        if shape is None:
            raise ValueError("Input MPCTensor must have valid shape.")

        ring_size = session.ring_size
        ring_bits = get_nr_bits(ring_size)
        msb_idx = ring_bits - 1  # index of msb

        share_ptrs_msb: List[ReplicatedSharedTensor] = []
        share_ptrs_wrap: List[ReplicatedSharedTensor] = []

        for share in a.share_ptrs:
            share_ptrs_msb.append(share.bit_extraction(msb_idx))
            share_ptrs_wrap.append(share << 1)

        msb = MPCTensor(shares=share_ptrs_msb, session=session, shape=shape)
        wrap_lshift = MPCTensor(shares=share_ptrs_wrap, session=session, shape=shape)

        wrap = Falcon.wrap(wrap_lshift)

        result = msb + wrap + 1

        return result

    @staticmethod
    def relu(a: MPCTensor) -> MPCTensor:
        """Computer ReLU on input MPCTensor.

        Args:
            a (MPCTensor): input MPCTensor.

        Returns:
            result (MPCTensor): ReLU of input tensor.

        Raises:
            ValueError: If the input tensor does not have a valid shape.
        """
        shape = a.shape
        tensor_type = a.session.tensor_type
        if shape is None:
            raise ValueError("Shape must be provided for ReLU.")

        Falcon.drelu(a)

        torch.zeros(shape).type(tensor_type)
        result = ""
        result = result + "1"

        return result
