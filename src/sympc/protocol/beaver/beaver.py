"""Beaver Triples Protocol.

D. Beaver. *Efficient multiparty protocols using circuit randomization*.
In J. Feigenbaum, editor, CRYPTO, volume **576** of Lecture Notes in
Computer Science, pages 420–432. Springer, 1991.
"""


# stdlib
import operator
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

# third party
import torch
import torchcsprng as csprng  # type: ignore

from sympc.config import Config
from sympc.session import Session
from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get
from sympc.store.exceptions import EmptyPrimitiveStore
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor import ShareTensor
from sympc.utils import count_wraps

ttp_generator = csprng.create_random_device_generator()

""" Those functions should be executed by the Trusted Party """


def _get_triples(
    op_str: str,
    nr_parties: int,
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    session: Session,
    ring_size: Optional[int] = None,
    config: Optional[Config] = None,
    **kwargs: Dict[Any, Any],
) -> List[Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]]:
    """Get triples.

    The Trusted Third Party (TTP) or Crypto Provider should provide this triples Currently,
    the one that orchestrates the communication provides those triples.".

    Args:
        op_str (str): Operator string.
        nr_parties (int): Number of parties
        a_shape (Tuple[int]): Shape of a from beaver triples protocol.
        b_shape (Tuple[int]): Shape of b part from beaver triples protocol.
        session (Session) : Session to generate the triples for.
        ring_size (int) : Ring Size of the triples to generate.
        config (Config) : The configuration(base,precision) of the shares to generate.
        kwargs: Arbitrary keyword arguments for commands.

    Returns:
        List[List[3 x List[ShareTensor, ShareTensor, ShareTensor]]]:
        The generated triples a,b,c for each party.

    Raises:
        ValueError: If the triples are not consistent.
        ValueError: If the share class is invalid.
    """
    from sympc.protocol import Falcon

    config = Config(encoder_precision=0)
    if op_str in ["conv2d", "conv_transpose2d"]:
        cmd = getattr(torch, op_str)
    else:
        cmd = getattr(operator, op_str)

    if session.protocol.share_class == ShareTensor:
        a_rand = torch.empty(size=a_shape, dtype=torch.long).random_(
            generator=ttp_generator
        )
        a_shares = MPCTensor.generate_shares(
            secret=a_rand,
            nr_parties=nr_parties,
            tensor_type=torch.long,
            config=config,
        )

        b_rand = torch.empty(size=b_shape, dtype=torch.long).random_(
            generator=ttp_generator
        )
        b_shares = MPCTensor.generate_shares(
            secret=b_rand,
            nr_parties=nr_parties,
            tensor_type=torch.long,
            config=config,
        )

        c_val = cmd(a_rand, b_rand, **kwargs)
        c_shares = MPCTensor.generate_shares(
            secret=c_val, nr_parties=nr_parties, tensor_type=torch.long, config=config
        )
    elif session.protocol.share_class == ReplicatedSharedTensor:

        if ring_size is None:
            ring_size = session.ring_size
        if config is None:
            config = session.config

        a_ptrs: List = []
        b_ptrs: List = []
        for session_ptr in session.session_ptrs:
            a_ptrs.append(
                session_ptr.prrs_generate_random_share(a_shape, str(ring_size))
            )
            b_ptrs.append(
                session_ptr.prrs_generate_random_share(b_shape, str(ring_size))
            )

        a = MPCTensor(shares=a_ptrs, session=session, shape=a_shape)
        b = MPCTensor(shares=b_ptrs, session=session, shape=b_shape)
        c = Falcon.mul_semi_honest(
            a, b, session, op_str, ring_size, config, reshare=True, **kwargs
        )

        a_shares = [share.get_copy() for share in a.share_ptrs]
        b_shares = [share.get_copy() for share in b.share_ptrs]
        c_shares = [share.get_copy() for share in c.share_ptrs]

        shares_sum = ReplicatedSharedTensor.shares_sum
        a_val = shares_sum([a_shares[0].shares[0]] + a_shares[1].shares, ring_size)
        b_val = shares_sum([b_shares[0].shares[0]] + b_shares[1].shares, ring_size)
        c_val = shares_sum([c_shares[0].shares[0]] + c_shares[1].shares, ring_size)

        op = ReplicatedSharedTensor.get_op(ring_size, op_str)

        if (c_val != op(a_val, b_val)).all():
            raise ValueError("Computation aborted:Invalid Triples")
    else:
        raise ValueError("Invalid share class.")

    # We are always creating an instance
    triple_sequential = [(a_shares, b_shares, c_shares)]

    """
    Example -- for n_instances=2 and n_parties=2:
    For Beaver Triples the "res" would look like:
    res = [
        ([a0_sh_p0, a0_sh_p1], [b0_sh_p0, b0_sh_p1], [c0_sh_p0, c0_sh_p1]),
        ([a1_sh_p0, a1_sh_p1], [b1_sh_p0, b1_sh_p1], [c1_sh_p0, c1_sh_p1])
    ]

    We want to send to each party the values they should hold:
    primitives = [
        [[a0_sh_p0, b0_sh_p0, c0_sh_p0], [a1_sh_p0, b1_sh_p0, c1_sh_p0]], # (Row 0)
        [[a0_sh_p1, b0_sh_p1, c0_sh_p1], [a1_sh_p1, b1_sh_p1, c1_sh_p1]]  # (Row 1)
    ]

    The first party (party 0) receives Row 0 and the second party (party 1) receives Row 1
    """

    triple = list(map(list, zip(*map(lambda x: map(list, zip(*x)), triple_sequential))))

    return triple


""" Beaver Operations defined for Multiplication """


@register_primitive_generator("beaver_mul")
def get_triples_mul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """Get the beaver triples for the multiplication operation.

    Args:
        *args (List[ShareTensor]): Named arguments of :func:`beaver.__get_triples`.
        **kwargs (List[ShareTensor]): Keyword arguments of :func:`beaver.__get_triples`.

    Returns:
        Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]: The generated triples a,b,c
        for the mul operation.
    """
    return _get_triples("mul", *args, **kwargs)


@register_primitive_store_add("beaver_mul")
def mul_store_add(
    store: Dict[Any, Any],
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:
    """Add the primitives required for the "mul" operation to the CryptoStore.

    Arguments:
        store (Dict[Any, Any]): the CryptoStore
        primitives (Iterable[Any]): the list of primitives
        a_shape (Tuple[int]): the shape of the first operand
        b_shape (Tuple[int]): the shape of the second operand
    """
    config_key = f"beaver_mul_{a_shape}_{b_shape}"
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_mul")
def mul_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "mul" operation.

    Args:
        store (Dict[Tuple[int, int], List[Any]]): The CryptoStore.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "mul" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_mul_{tuple(a_shape)}_{tuple(b_shape)}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Matrix Multiplication """


@register_primitive_generator("beaver_matmul")
def get_triples_matmul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """Get the beaver triples for the matmul  operation.

    Args:
        *args (List[ShareTensor]): Named arguments of :func:`beaver.__get_triples`.
        **kwargs (List[ShareTensor]): Keyword arguments of :func:`beaver.__get_triples`.

    Returns:
        Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]: The generated triples a,b,c
        for the matmul operation.
    """
    return _get_triples("matmul", *args, **kwargs)


@register_primitive_store_add("beaver_matmul")
def matmul_store_add(
    store: Any,
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:
    """Add the primitives required for the "matmul" operation to the CryptoStore.

    Args:
        store (Any): The CryptoStore.
        primitives (Iterable[Any]): The list of primitives
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.

    """
    config_key = f"beaver_matmul_{a_shape}_{b_shape}"
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_matmul")
def matmul_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "matmul" operation.

    Args:
        store (Dict[Tuple[int, int], List[Any]]): The CryptoStore.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "matmul" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_matmul_{tuple(a_shape)}_{tuple(b_shape)}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Convolution 2D """


@register_primitive_generator("beaver_conv2d")
def get_triples_conv2d(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """Get the beaver triples for the conv2d operation.

    Args:
        *args: Arguments for _get_triples.
        **kwargs: Keyword arguments for _get_triples.

    Returns:
        Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]: The generated
        triples a,b,c for each party.
    """
    return _get_triples("conv2d", *args, **kwargs)


@register_primitive_store_add("beaver_conv2d")
def conv2d_store_add(
    store: Any,
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:
    """Add the primitives required for the "conv2d" operation to the CryptoStore.

    Args:
        store (Any): The CryptoStore.
        primitives (Iterable[Any]): The list of primitives.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
    """
    config_key = f"beaver_conv2d_{a_shape}_{b_shape}"
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_conv2d")
def conv2d_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "conv2d" operation.

    Args:
        store: the CryptoStore
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "conv2d" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_conv2d_{tuple(a_shape)}_{tuple(b_shape)}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Convolution Transpose 2D """


@register_primitive_generator("beaver_conv_transpose2d")
def get_triples_transpose2d(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """Get the beaver triples for the conv_transpose2d operation.

    Args:
        *args: Arguments for _get_triples.
        **kwargs: Keyword arguments for _get_triples.

    Returns:
        Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]: The generated
        triples a,b,c for each party.
    """
    return _get_triples("conv_transpose2d", *args, **kwargs)


@register_primitive_store_add("beaver_conv_transpose2d")
def conv_transpose2d_store_add(
    store: Any,
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:
    """Add the primitives required for the "conv_transpose2d" operation to the CryptoStore.

    Args:
        store (Any): The CryptoStore.
        primitives (Iterable[Any]): The list of primitives.
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
    """
    config_key = f"beaver_conv_transpose2d_{a_shape}_{b_shape}"
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_conv_transpose2d")
def conv_transpose2d_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    remove: bool = True,
) -> Any:
    """Retrieve the primitives from the CryptoStore.

    Those are needed for executing the "conv_transpose2d" operation.

    Args:
        store: the CryptoStore
        a_shape (Tuple[int]): The shape of the first operand.
        b_shape (Tuple[int]): The shape of the second operand.
        remove (bool): True if the primitives should be removed from the store.

    Returns:
        Any: The primitives required for the "conv2d" operation.

    Raises:
        EmptyPrimitiveStore: If no primitive in the store for config_key.
    """
    config_key = f"beaver_conv_transpose2d_{tuple(a_shape)}_{tuple(b_shape)}"

    try:
        primitives = store[config_key]
    except KeyError:
        raise EmptyPrimitiveStore(f"{config_key} does not exists in the store")

    try:
        primitive = primitives[0]
    except Exception:
        raise EmptyPrimitiveStore(f"No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Counting the Wrap-Arounds """


@register_primitive_generator("beaver_wraps")
def count_wraps_rand(
    nr_parties: int, shape: Tuple[int]
) -> Tuple[List[ShareTensor], List[ShareTensor]]:
    """Count wraps random.

    The Trusted Third Party (TTP) or Crypto provider should generate:

    - a set of shares for a random number
    - a set of shares for the number of wraparounds for that number

    Those shares are used when doing a public division, such that the
    end result would be the correct one.

    Args:
        nr_parties (int): Number of parties
        shape (Tuple[int]): The shape for the random value

    Returns:
        List[List[List[ShareTensor, ShareTensor]]: a list of instaces with the shares
        for a random integer value and shares for the number of wraparounds that are done when
        reconstructing the random value
    """
    rand_val = torch.empty(size=shape, dtype=torch.long).random_(
        generator=ttp_generator
    )

    config = Config(encoder_precision=0)
    r_shares = MPCTensor.generate_shares(
        secret=rand_val, nr_parties=nr_parties, tensor_type=torch.long, config=config
    )
    wraps = count_wraps([share.tensor for share in r_shares])

    theta_r_shares = MPCTensor.generate_shares(
        secret=wraps, nr_parties=nr_parties, tensor_type=torch.long, config=config
    )

    # We are always creating only an instance
    primitives_sequential = [(r_shares, theta_r_shares)]

    primitives = list(
        map(list, zip(*map(lambda x: map(list, zip(*x)), primitives_sequential)))
    )

    return primitives
