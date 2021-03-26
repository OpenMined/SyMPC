"""Function Secret Sharing Protocol.

ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing
arXiv:2006.04593 [cs.LG]
"""
# stdlib
import math
import multiprocessing
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

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
from sympc.utils import parallel_execution

ttp_generator = csprng.create_random_device_generator()

λ = 127  # security parameter
n = 32  # bit precision
N = 4  # byte precision
λs = math.ceil(λ / 64)  # how many int64 are needed to store λ, here 2
if λs != 2:
    raise ValueError("Check the value of security parameter")

# internal codes
EQ = 0
COMP = 1

# number of processes
N_CORES = multiprocessing.cpu_count()
dpf = sycret.EqFactory(n_threads=N_CORES)
dif = sycret.LeFactory(n_threads=N_CORES)


def keygen(n_values: int, op: str):
    """Run FSS keygen in parallel to accelerate the offline part of the protocol.

    Args:
        n_values (int): Number of primitives to generate.
        op (str): Operator eq or comp <=> DPF or DIF.

    Returns:
        noqa: DAR201

    Raises:
        ValueError: if "op" is not "eq" or "comp".
    """
    if op == "eq":
        return DPF.keygen(n_values=n_values)
    if op == "comp":
        return DIF.keygen(n_values=n_values)

    raise ValueError(f"{op} is an unsupported operation.")


def fss_op(x1: ShareTensor, x2: ShareTensor, op="eq"):
    """Define the workflow for a binary operation using Function Secret Sharing.

    Currently supported operand are = & <=, respectively corresponding to
        op = 'eq' and 'comp'

    Args:
        x1 (ShareTensor): First AST.
        x2 (ShareTensor): Second AST.
        op: Type of operation to perform, should be 'eq' or 'comp'. Defaults to eq.

    Returns:
        MPCTensor: Shares of the comparison.
    """
    assert not th.cuda.is_available()  # nosec

    # FIXME: Better handle the case where x1 or x2 is not a MPCTensor. For the moment
    # FIXME: we cast it into a MPCTensor at the expense of extra communication
    session = x1.session
    dtype = session.tensor_type

    shape = MPCTensor._get_shape("sub", x1.shape, x2.shape)
    n_values = shape.numel()

    CryptoPrimitiveProvider.generate_primitives(
        f"fss_{op}",
        sessions=session.session_ptrs,
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

    response = MPCTensor(session=session, shares=shares)
    response.shape = shape
    return response


# share level
def mask_builder(
    session: Session, x1: ShareTensor, x2: ShareTensor, op: str
) -> ShareTensor:
    """Add the share of alpha (the mask) that is held in the crypto store to the 
    difference x1 - x2.
    
    As we aim at comparing x1 <= x2, we actually compare x1 - x2 <= 0 and we hide
    x1 - x2 with alpha that is a random mask.

    Args:
        session (Session): MPC Session.
        x1 (ShareTensor): First AST.
        x2 (ShareTensor): Second AST.
        op (str): Type of operation to perform.

    Returns:
        ShareTensor

    """
    x = x1 - x2

    keys = session.crypto_store.get_primitives_from_store(
        f"fss_{op}", nr_instances=x.numel(), remove=False
    )

    alpha = np.frombuffer(np.ascontiguousarray(keys[:, 0:N]), dtype=np.uint32)

    x.tensor += th.tensor(alpha.astype(np.int64)).reshape(x.shape)

    return x


# share level
def evaluate(session: Session, b, x_masked, op, dtype="long") -> ShareTensor:
    """Evaluate the FSS protocol on the masked and public input `x_masked`

    Args:
        session (Session): MPC Session.
        b: noqa: DAR101
        x_masked: noqa: DAR101
        op: noqa: DAR101
        dtype: noqa: DAR101

    Returns:
        ShareTensor: TODO.

    Raises:
        ValueError: If "op" is not "eq" or "comp".
    """
    if op == "eq":
        return eq_evaluate(session, b, x_masked)
    elif op == "comp":
        return comp_evaluate(session, b, x_masked, dtype=dtype)
    else:
        raise ValueError


# process level
def eq_evaluate(session: Session, b, x_masked) -> ShareTensor:
    """TODO: Add docstring.

    Args:
        session (Session): MPC Session.
        b: noqa: DAR101
        x_masked: noqa: DAR101

    Returns:
        ShareTensor
    """
    numel = x_masked.numel()
    keys = session.crypto_store.get_primitives_from_store(
        "fss_eq", nr_instances=numel, remove=True
    )

    result = DPF.eval(b.numpy().item(), x_masked.numpy(), keys)

    share_result = ShareTensor(
        data=th.tensor(result), session=session
    )  # TODO add dtype like in comp_evaluate

    return share_result


# process level
def comp_evaluate(session: Session, b, x_masked, dtype=None) -> ShareTensor:
    """TODO: Add docstring.

    Args:
        session (Session): MPC Session.
        b: noqa: DAR101
        x_masked: noqa: DAR101
        dtype: noqa: DAR101

    Returns:
        ShareTensor: TODO
    """
    numel = x_masked.numel()
    keys = session.crypto_store.get_primitives_from_store(
        "fss_comp", nr_instances=numel, remove=True
    )

    result_share = DIF.eval(b.numpy().item(), x_masked.numpy(), keys)

    dtype_options = {None: th.long, "int": th.int32, "long": th.long}
    result = th.tensor(result_share, dtype=dtype_options[dtype])

    share_result = ShareTensor(data=result, session=session)

    return share_result


class DPF:
    """Distributed Point Function - used for equality."""

    # third party
    from sycret.fss import FSSFactory

    @staticmethod
    def keygen(n_values=1):
        """Sycret DPF keygen.

        Args:
            n_values (int): Number of values. Defaults to 1

        Returns:
            TODO: Add return type
        """
        return dpf.keygen(n_values=n_values)

    @staticmethod
    def eval(b, x, k_b):
        """Sycret DPF eval.

        Args:
            b: noqa: DAR101
            x: noqa: DAR101
            k_b: noqa: DAR101

        Returns:
            noqa: DAR201
        """
        original_shape = x.shape
        x = x.reshape(-1)
        flat_result = dpf.eval(b, x, k_b)
        return flat_result.astype(np.int32).astype(np.int64).reshape(original_shape)


class DIF:
    """Distributed Interval Function - used for comparison."""

    @staticmethod
    def keygen(n_values=1):
        """Sycret DIF keygen.

        Args:
            n_values (int): Number of values. Defaults to 1

        Returns:
            TODO: Add return type
        """
        return dif.keygen(n_values=n_values)

    @staticmethod
    def eval(b, x, k_b):
        """Sycret DIF eval.

        Args:
            b: noqa: DAR101
            x: noqa: DAR101
            k_b: noqa: DAR101

        Returns:
            # noqa: DAR201
        """
        # x = x.astype(np.uint64)
        original_shape = x.shape
        x = x.reshape(-1)
        flat_result = dif.eval(b, x, k_b)
        return flat_result.astype(np.int32).astype(np.int64).reshape(original_shape)


class FSS(metaclass=Protocol):
    """Function Secret Sharing."""

    @staticmethod
    def eq(x1: ShareTensor, x2: ShareTensor):
        """Equal operator.

        Args:
            x1 (ShareTensor): First AST.
            x2 (ShareTensor): Second AST.

        Returns:
            MPCTensor: Shares of the equality.
        """
        return fss_op(x1, x2, "eq")

    @staticmethod
    def le(x1, x2):
        """Lower equal operator.

        Args:
            x1 (ShareTensor): First AST.
            x2 (ShareTensor): Second AST.

        Returns:
            MPCTensor: Shares of the comparison.
        """
        return fss_op(x1, x2, "comp")


""" Register Crypto Store capabilities for FSS """


def _ensure_fss_store(store: Dict[Any, Any]):
    for fss_key in ["fss_eq", "fss_comp"]:
        if fss_key not in store.keys():
            store[fss_key] = [[]]


def _generate_primitive(op: str, n_values: int) -> List[Any]:
    primitives = keygen(n_values, op=op)
    return [th.tensor(p) for p in primitives]


@register_primitive_generator("fss_eq")
def generate_primitive(n_values: int) -> List[Any]:
    """Generate primitives for the FSS equality protocol.

    Args:
        n_values (int): Number of values.

    Returns:
        List[Any]: TODO.
    """
    return _generate_primitive("eq", n_values)


@register_primitive_generator("fss_comp")
def generate_primitive(n_values: int) -> List[Any]:
    """TODO: Add docstring.

    Args:
        n_values (int): Number of values.

    Returns:
        List[Any]: TODO.
    """  # Seems almost the same than previous one, can they be unified?
    return _generate_primitive("comp", n_values)


def _add_primitive(
    op: str,
    store: Dict[Any, Any],
    primitives: Iterable[Any],
):
    """Add FSS primitives to the crypto store.

    Args:
        op (str): Operator.
        store (Dict[Any,Any]): TODO
        primitives (Iterable[Any]): TODO
    """
    _ensure_fss_store(store)
    current_primitives = store[op]

    primitives = np.array(primitives)

    if len(current_primitives) == 0 or len(current_primitives[0]) == 0:
        store[op] = [primitives]
    else:
        # This branch never happens with on-the-fly primitives
        current_primitives.append(primitives)


@register_primitive_store_add("fss_eq")
def add_primitive(
    store: Dict[Any, Any],
    primitives: Iterable[Any],
) -> None:
    """Add FSS primitives for equality to the crypto store.

    Args:
        store (Dict[Any,Any]): TODO
        primitives (Iterable[Any]): TODO

    Returns:
        # noqa: DAR201
    """
    return _add_primitive("fss_eq", store, primitives)


@register_primitive_store_add("fss_comp")
def add_primitive(store: Dict[Any, Any], primitives: Iterable[Any]):
    """Add FSS primitives for comparison to the crypto store.

    Args:
        store (Dict[Any,Any]): TODO
        primitives (Iterable[Any]): TODO

    Returns:
        # noqa: DAR201
    """  # Seems almost the same than previous one, can they be unified?
    return _add_primitive("fss_comp", store, primitives)


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


@register_primitive_store_get("fss_eq")  # noqa
def get_primitive(
    store: Dict[Tuple[int, int], List[Any]],
    nr_instances: int,
    remove: bool = True,
    **kwargs,
) -> Any:  # noqa
    return _get_primitive("fss_eq", store, nr_instances, remove, **kwargs)


@register_primitive_store_get("fss_comp")
def get_primitive(
    store: Dict[Tuple[int, int], List[Any]],
    nr_instances: int,
    remove: bool = True,
    **kwargs,
) -> Any:  # noqa
    return _get_primitive("fss_comp", store, nr_instances, remove, **kwargs)
