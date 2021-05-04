"""Applications that make use of primitive MPC operations."""

# stdlib
from typing import List

# third party
# third-party
import torch

from sympc.tensor import MPCTensor


def sort(input_list: List[MPCTensor], ascending: bool = True) -> List[MPCTensor]:
    """Takes a list of MPCTensors and sorts them in ascending/desending order.

    Args:
        input_list (List[MPCTensor]): Takes a list of MPCTensor
        ascending (bool): If list has to sorted in ascending/descending order

    Returns:
        List[MPC_tensor]: Sorted list of MPCTensors
    """

    # Checks if the list of MPCTensors are of length 1
    if not all((item.shape == torch.Size([1])) for item in input_list):

        raise ValueError(
            "Invalid dimension. All MPCTensors should have an 1-dimensional secret."
        )

    if len(input_list) > 1:
        mid = len(input_list) // 2
        left = input_list[:mid]
        right = input_list[mid:]

        # Recursive call on each half
        sort(left)
        sort(right)

        # Two iterators for traversing the two halves
        i = 0
        j = 0

        # Iterator for the main list
        k = 0

        while i < len(left) and j < len(right):
            if (left[i] < right[j]).reconstruct():
                # The value from the left half has been used
                input_list[k] = left[i]
                # Move the iterator forward
                i += 1
            else:
                input_list[k] = right[j]
                j += 1
            # Move to the next slot
            k += 1

        # For all the remaining values
        while i < len(left):
            input_list[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            input_list[k] = right[j]
            j += 1
            k += 1

    if not ascending:

        return input_list[::-1]

    return input_list
