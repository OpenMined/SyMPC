"""
The Beaver Triples
"""

# stdlib
import operator
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from typing import Iterable

# third party
import torch
import torchcsprng as csprng  # type: ignore

from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import count_wraps
from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get

ttp_generator = csprng.create_random_device_generator()

""" Those functions should be executed by the Trusted Party """


def _get_triples(
    op_str: str, nr_parties: int, a_shape: Tuple[int], b_shape: Tuple[int]
) -> Tuple[Tuple[ShareTensor, ShareTensor, ShareTensor]]:
    """
    The Trusted Third Party (TTP) or Crypto Provider should provide this triples
    Currently, the one that orchestrates the communication provides those triples.
    """

    a_rand = torch.empty(size=a_shape, dtype=torch.long).random_(
        generator=ttp_generator
    )
    a = ShareTensor(data=a_rand, encoder_precision=0)
    a_shares = MPCTensor.generate_shares(a, nr_parties, torch.long)

    b_rand = torch.empty(size=b_shape, dtype=torch.long).random_(
        generator=ttp_generator
    )
    b = ShareTensor(data=b_rand, encoder_precision=0)
    b_shares = MPCTensor.generate_shares(b, nr_parties, torch.long)

    cmd = getattr(operator, op_str)

    c_val = cmd(a_rand, b_rand)
    c = ShareTensor(data=c_val, encoder_precision=0)
    c_shares = MPCTensor.generate_shares(c, nr_parties, torch.long)

    return a_shares, b_shares, c_shares


""" Beaver Operations defined for Multiplication """


@register_primitive_generator("beaver_mul")
def get_triples_mul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """ Get the beaver triples for the matmul operation"""
    return _get_triples("mul", *args, **kwargs)


@register_primitive_store_add("beaver_mul")
def mul_store_add(
    store: Dict[Any, Any],
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:
    config_key = (a_shape, b_shape)
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_mul")
def mul_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    remove: bool = True,
) -> Any:
    config_key = (a_shape, b_shape)
    primitives = store[config_key]

    try:
        primitive = primitives[0]
    except _:
        raise ValueError("No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Matrix Multiplication """


@register_primitive_generator("beaver_matmul")
def get_triples_matmul(
    *args: List[Any], **kwargs: Dict[Any, Any]
) -> Tuple[List[ShareTensor], List[ShareTensor], List[ShareTensor]]:
    """ Get the beaver triples for the mul operation """
    return _get_triples("matmul", *args, **kwargs)


@register_primitive_store_add("beaver_matmul")
def matmul_store_add(
    store: Any,
    primitives: Iterable[Any],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
) -> None:

    config_key = (a_shape, b_shape)
    if config_key in store:
        store[config_key].extend(primitives)
    else:
        store[config_key] = primitives


@register_primitive_store_get("beaver_matmul")
def matmul_store_get(
    store: Dict[Tuple[int, int], List[Any]],
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    remove: bool = True,
) -> Any:
    config_key = (a_shape, b_shape)
    primitives = store[config_key]

    try:
        primitive = primitives[0]
    except _:
        raise ValueError("No primitive in the store for {config_key}")

    if remove:
        del primitives[0]

    return primitive


""" Beaver Operations defined for Counting the Wrap-Arounds """


@register_primitive_generator("beaver_wraps")
def count_wraps_rand(
    nr_parties: int, shape: Tuple[int]
) -> Tuple[List[ShareTensor], List[ShareTensor]]:
    """
    The Trusted Third Party (TTP) or Crypto Provider should generate
    - a set of shares for a random number
    - a set of shares for the number of wraparounds for that number

    Those shares are used when doing a public division, such that the
    end result would be the correct one.
    """
    rand_val = torch.empty(size=shape, dtype=torch.long).random_(
        generator=ttp_generator
    )
    r = ShareTensor(data=rand_val, encoder_precision=0)

    r_shares = MPCTensor.generate_shares(r, nr_parties, torch.long)
    wraps = count_wraps(r_shares)

    theta_r = ShareTensor(data=wraps, encoder_precision=0)
    theta_r.tensor = wraps

    theta_r_shares = MPCTensor.generate_shares(theta_r, nr_parties, torch.long)

    return r_shares, theta_r_shares
