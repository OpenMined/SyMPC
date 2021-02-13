# stdlib
from functools import lru_cache
from typing import List
from typing import Union

# third party
# TODO: Remove this
import torch
import torchcsprng as csprng  # type: ignore

RING_SIZE_TO_TYPE = {
    2 ** 1: torch.bool,
    2 ** 8: torch.int8,
    2 ** 16: torch.int16,
    2 ** 32: torch.int32,
    2 ** 64: torch.int64,
}


def count_wraps(share_list: List[torch.tensor]) -> torch.Tensor:
    """Count the number of overflows and underflows that would happen if we
    reconstruct the original value

    This function is taken from the CrypTen project.
    CrypTen repository: https://github.com/facebookresearch/CrypTen

    Args:
        share_list (List[ShareTensor]): List of the shares

    Returns:
        The number of wraparounds
    """

    res = torch.zeros(size=share_list[0].size(), dtype=torch.long)
    prev_share = share_list[0]
    for cur_share in share_list[1:]:
        next_share = cur_share + prev_share

        # If prev and current shares are negative,
        # but the result is positive then is an underflow
        res -= ((prev_share < 0) & (cur_share < 0) & (next_share > 0)).long()

        # If prev and current shares are positive,
        # but the result is positive then is an overflow
        res += ((prev_share > 0) & (cur_share > 0) & (next_share < 0)).long()
        prev_share = next_share

    return res


@lru_cache(maxsize=len(RING_SIZE_TO_TYPE))
def get_type_from_ring(ring_size: int) -> torch.dtype:
    """Get the type of a tensor given a ring size/field
    :return: the type of tensors
    :rtype: torch:dtype
    """
    if ring_size not in RING_SIZE_TO_TYPE:
        raise ValueError(f"Ring size should be in {RING_SIZE_TO_TYPE.keys()}")

    return RING_SIZE_TO_TYPE[ring_size]


def get_new_generator(seed: int) -> torch.Generator:
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


@lru_cache
def get_nr_bits(ring_size: int) -> int:
    return (ring_size - 1).bit_length()


def decompose(tensor: torch.Tensor, ring_size: int, shape=None) -> torch.Tensor:
    """Decompose a tensor into its binary representation."""

    tensor_type = get_type_from_ring(ring_size)

    nr_bits = get_nr_bits(ring_size)
    powers = torch.arange(nr_bits, dtype=tensor_type)

    if shape is None:
        shape = tensor.shape

    for _ in range(len(shape)):
        powers = powers.unsqueeze(0)

    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = torch.fmod((tensor / moduli.type_as(tensor)), 2)
    print("Tensor wtf", tensor)
    return tensor
