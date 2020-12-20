"""
Util functions used when doing SMPC
"""

from functools import lru_cache
from typing import Union

import torch
import torchcsprng as csprng  # type: ignore

RING_SIZE_TO_TYPE = {
    2 ** 1: torch.bool,
    2 ** 8: torch.int8,
    2 ** 16: torch.int16,
    2 ** 32: torch.int32,
    2 ** 64: torch.int64,
}


@lru_cache(maxsize=len(RING_SIZE_TO_TYPE))
def get_type_from_ring(ring_size: int) -> torch.dtype:
    """Get the type of a tensor given a ring size/field
    :return: the type of tensors
    :rtype: torch:dtype
    """
    if ring_size not in RING_SIZE_TO_TYPE:
        raise ValueError(f"Ring size should be in {RING_SIZE_TO_TYPE.keys()}")

    return RING_SIZE_TO_TYPE[ring_size]


def get_generator(seed: int) -> torch.Generator:
    """Get a generator that is initialzed with the specified seed
    :return: a generator
    :rtype: torch.Generator
    """
    # TODO: this is not crypto secure, but it lets you add a seed
    return csprng.create_mt19937_generator(seed=seed)


def generate_random_element(
    tensor_type: torch.dtype,
    generator: torch.Generator,
    shape: Union[tuple, torch.Size],
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a new "random" tensor

    :return: a random tensor using a specific generator
    :rtype: torch.Tensor
    """
    return torch.empty(size=shape, dtype=tensor_type, device=device).random_(
        generator=generator
    )
