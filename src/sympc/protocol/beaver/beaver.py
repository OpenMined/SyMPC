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
from typing import Tuple

# third party
import torch
import torchcsprng as csprng  # type: ignore

from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get
from sympc.store.exceptions import EmptyPrimitiveStore
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor

ttp_generator = csprng.create_random_device_generator()

""" Those functions should be executed by the Trusted Party """


def _get_triples(
    op_str: str,
    nr_parties: int,
    a_shape: Tuple[int],
    b_shape: Tuple[int],
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
        kwargs: Arbitrary keyword arguments for commands.


    Returns:
        List[List[3 x List[ShareTensor, ShareTensor, ShareTensor]]]:
        The generated triples a,b,c for each party.
    """
    a_rand = torch.empty(size=a_shape, dtype=torch.long).random_(
        generator=ttp_generator
    )
    a_shares = MPCTensor.generate_shares(
        secret=a_rand,
        nr_parties=nr_parties,
        tensor_type=torch.long,
        encoder_precision=0,
    )

    b_rand = torch.empty(size=b_shape, dtype=torch.long).random_(
        generator=ttp_generator
    )
    b_shares = MPCTensor.generate_shares(
        secret=b_rand,
        nr_parties=nr_parties,
        tensor_type=torch.long,
        encoder_precision=0,
    )

    if op_str in ["conv2d", "conv_transpose2d"]:
        cmd = getattr(torch, op_str)
    else:
        cmd = getattr(operator, op_str)

    c_val = cmd(a_rand, b_rand, **kwargs)
    c_shares = MPCTensor.generate_shares(
        secret=c_val, nr_parties=nr_parties, tensor_type=torch.long, encoder_precision=0
    )

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
