"""Function Secret Sharing Protocol.

ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing
arXiv:2006.04593 [cs.LG]
"""
# stdlib
import multiprocessing
import os
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
import warnings

# third party
import numpy as np
import sycret
import torch as th
import torchcsprng as csprng  # type: ignore

from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.store import CryptoPrimitiveProvider
from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.tensor.tensor import SyMPCTensor
from sympc.utils import parallel_execution

ttp_generator = csprng.create_random_device_generator()

λ = 127  # security parameter
n = 32  # bit precision

# number of processes
N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)


# share level
def mask_builder(
    session: Session, x1: ShareTensor, x2: ShareTensor, op: str
) -> ShareTensor:
    """Mask the private inputs.

    Add the share of alpha (the mask) that is held in the crypto store to
    the difference x1 - x2.
    As we aim at comparing x1 <= x2, we actually compare x1 - x2 <= 0 and we hide
    x1 - x2 with alpha that is a random mask.

    Args:
        session (Session): MPC Session.
        x1 (ShareTensor): Share of the first private value.
        x2 (ShareTensor): Share of the second private value.
        op (str): Type of operation to perform (eq or comp).

    Returns:
        ShareTensor: share of the masked input
    """
    x = x1 - x2

    keys = session.crypto_store.get_primitives_from_store(
        f"fss_{op}", nr_instances=x.numel(), remove=False
    )

    n_bytes = n // 8  # keys contains bytes not bits
    alpha = np.frombuffer(np.ascontiguousarray(keys[:, 0:n_bytes]), dtype=np.uint32)

    x.tensor += th.tensor(alpha.astype(np.int64)).reshape(x.shape)

    return x


# share level
def evaluate(session: Session, b, x_masked, op, dtype="long") -> ShareTensor:
    """Evaluate the FSS protocol on the masked and public input `x_masked`.

    Args:
        session (Session): MPC Session
        b: rank of the evaluator running this function
        x_masked: the public input created by masking the private input
        op: the type of operation (eq or comp)
        dtype: the type of the shares (int or long)

    Returns:
        ShareTensor: A share of the result of the FSS protocol.
    """
    numel = x_masked.numel()
    keys = session.crypto_store.get_primitives_from_store(
        f"fss_{op}", nr_instances=numel, remove=True
    )

    b = b.numpy().item()
    original_shape = x_masked.shape
    x_masked = x_masked.numpy().reshape(-1)

    if op == "eq":
        flat_result = dpf.eval(b, x_masked, keys)
    elif op == "comp":
        flat_result = dif.eval(b, x_masked, keys)

    result_share = flat_result.astype(np.int32).astype(np.int64).reshape(original_shape)

    dtype_options = {None: th.long, "int": th.int32, "long": th.long}
    result = th.tensor(result_share, dtype=dtype_options[dtype])

    share_result = ShareTensor(
        data=result, session_uuid=session.uuid, config=session.config
    )

    return share_result


def fss_op(x1: MPCTensor, x2: MPCTensor, op="eq") -> MPCTensor:
    """Define the workflow for a binary operation using Function Secret Sharing.

    Currently supported operand are = & <=, respectively corresponding to
    op = 'eq' and 'comp'.

    Args:
        x1 (MPCTensor): First private value.
        x2 (MPCTensor): Second private value.
        op: Type of operation to perform, should be 'eq' or 'comp'. Defaults to eq.

    Returns:
        MPCTensor: Shares of the comparison.
    """
    if th.cuda.is_available():
        # FSS is currently not supported on GPU.
        # https://stackoverflow.com/a/62145307/8878627

        # When the CUDA_VISIBLE_DEVICES environment variable is not set,
        # CUDA is not used even if available. Hence, we default to None
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        warnings.warn("Temporarily disabling CUDA as FSS does not support it")
    else:
        cuda_visible_devices = None

    # FIXME: Better handle the case where x1 or x2 is not a MPCTensor. For the moment
    # FIXME: we cast it into a MPCTensor at the expense of extra communication
    session = x1.session

    shape = MPCTensor._get_shape("sub", x1.shape, x2.shape)
    n_values = shape.numel()

    CryptoPrimitiveProvider.generate_primitives(
        f"fss_{op}",
        session=session,
        g_kwargs={"n_values": n_values},
        p_kwargs={},
    )

    args = zip(session.session_ptrs, x1.share_ptrs, x2.share_ptrs)
    args = [list(el) + [op] for el in args]

    shares = parallel_execution(mask_builder, session.parties)(args)

    # TODO: don't do .reconstruct(), this should be done remotely between the evaluators
    mask_value = MPCTensor(shares=shares, session=session)
    mask_value = mask_value.reconstruct(decode=False) % 2 ** n

    # TODO: add dtype to args
    args = [
        (session.session_ptrs[i], th.IntTensor([i]), mask_value, op) for i in range(2)
    ]

    shares = parallel_execution(evaluate, session.parties)(args)

    response = MPCTensor(session=session, shares=shares, shape=shape)
    response.shape = shape

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    return response


class FSS(metaclass=Protocol):
    """Function Secret Sharing."""

    """ Used for Share Level static operations like distributing the shares."""
    share_class: SyMPCTensor = ShareTensor
    security_levels: List[str] = ["semi-honest"]

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
    def eq(x1: MPCTensor, x2: MPCTensor) -> MPCTensor:
        """Equal operator.

        Args:
            x1 (MPCTensor): First private value.
            x2 (MPCTensor): Second private value.

        Returns:
            MPCTensor: Shares of the equality.
        """
        return fss_op(x1, x2, "eq")

    @staticmethod
    def le(x1: MPCTensor, x2: MPCTensor) -> MPCTensor:
        """Lower equal operator.

        Args:
            x1 (MPCTensor): First private value.
            x2 (MPCTensor): Second private value.

        Returns:
            MPCTensor: Shares of the comparison.
        """
        return fss_op(x1, x2, "comp")

    @staticmethod
    def distribute_shares(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        """Forward the call to the tensor specific class.

        Args:
            *args (List[Any]): list of args to be forwarded
            **kwargs(Dict[str, Any): list of named args to be forwarded

        Returns:
            The result returned by the tensor specific distribute_shares method
        """
        return FSS.share_class.distribute_shares(*args, **kwargs)

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


""" Register Crypto Store capabilities for FSS """


def _ensure_fss_store(store: Dict[Any, Any]):
    """Create an empty store for FSS if it doesn't exist already.

    Args:
        store: the main crypto store
    """
    for fss_key in ["fss_eq", "fss_comp"]:
        if fss_key not in store.keys():
            store[fss_key] = [[]]


def _generate_primitive(op: str, n_values: int) -> List[Any]:
    """Generate FSS primitives.

    Args:
        op (str): type of operation (eq or comp)
        n_values (int): number of primitives to generate

    Returns:
        List[Any]: a pair of primitive keys

    Raises:
        ValueError: if the operation is not valid
    """
    if op == "eq":
        primitives = dpf.keygen(n_values=n_values)
    elif op == "comp":
        primitives = dif.keygen(n_values=n_values)
    else:
        raise ValueError(f"{op} is an FSS unsupported operation.")

    return [th.tensor(p) for p in primitives]


def _add_primitive(
    op: str,
    store: Dict[Any, Any],
    primitives: Iterable[Any],
):
    """Add FSS primitives to the crypto store.

    Args:
        op (str): type of operation (eq or comp)
        store (Dict[Any,Any]): the crypto store
        primitives (Iterable[Any]): the primitives to add to the store
    """
    _ensure_fss_store(store)
    current_primitives = store[op]

    primitives = np.array(primitives)

    if len(current_primitives) == 0 or len(current_primitives[0]) == 0:
        store[op] = [primitives]
    else:
        # This branch never happens with on-the-fly primitives
        current_primitives.append(primitives)


def _get_primitive(
    op: str,
    store: Dict[Tuple[int, int], List[Any]],
    nr_instances: int,
    remove: bool = True,
    **kwargs,
) -> Any:
    _ensure_fss_store(store)
    primitive_stack = store[op]

    available_instances = len(primitive_stack[0]) if len(primitive_stack) > 0 else -1

    if available_instances >= nr_instances:
        primitive = primitive_stack[0][0:nr_instances]
        if remove:
            # We throw the whole key array away, not just the keys we used
            del primitive_stack[0]
        return primitive
    else:
        raise ValueError(
            f"Not enough primitives for fss: {nr_instances} required, "
            f"but only {available_instances} available"
        )


@register_primitive_generator("fss_eq")
def generate_primitive(n_values: int) -> List[Any]:  # noqa
    return _generate_primitive("eq", n_values)


@register_primitive_generator("fss_comp")
def generate_primitive(n_values: int) -> List[Any]:  # noqa
    return _generate_primitive("comp", n_values)


@register_primitive_store_add("fss_eq")
def add_primitive(store: Dict[Any, Any], primitives: Iterable[Any]) -> None:  # noqa
    return _add_primitive("fss_eq", store, primitives)


@register_primitive_store_add("fss_comp")
def add_primitive(store: Dict[Any, Any], primitives: Iterable[Any]):  # noqa
    return _add_primitive("fss_comp", store, primitives)


@register_primitive_store_get("fss_eq")
def get_primitive(
    store: Dict[Any, Any], nr_instances: int, remove: bool = True, **kwargs
) -> Any:  # noqa
    return _get_primitive("fss_eq", store, nr_instances, remove, **kwargs)


@register_primitive_store_get("fss_comp")
def get_primitive(
    store: Dict[Any, Any], nr_instances: int, remove: bool = True, **kwargs
) -> Any:  # noqa
    return _get_primitive("fss_comp", store, nr_instances, remove, **kwargs)
