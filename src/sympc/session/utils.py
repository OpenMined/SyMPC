from functools import lru_cache
import torch

RING_SIZE_TO_TYPE = {2 ** 16: torch.int16, 2 ** 32: torch.int32, 2 ** 64: torch.int64}


@lru_cache(maxsize=len(RING_SIZE_TO_TYPE))
def get_type_from_ring(ring_size: int) -> torch.dtype:
    if ring_size not in RING_SIZE_TO_TYPE:
        raise ValueError(f"Ring size should be in {RING_SIZE_TO_TYPE.keys()}")

    return RING_SIZE_TO_TYPE[ring_size]
