"""Applications that make use of primitive MPC operations."""

# stdlib
from typing import List

# third party
import torch

from sympc.tensor import MPCTensor


def sort(input_list: List[MPCTensor], ascending: bool = True) -> List[MPCTensor]:
    """Takes a list of MPCTensors and sorts them in ascending/desending order using bubble sort.

    Args:
        input_list (List[MPCTensor]): Takes a list of MPCTensor
        ascending (bool): If list has to sorted in ascending/descending order

    Returns:
        List[MPC_tensor]: Sorted list of MPCTensors

    Raises:
        ValueError: If the list contains MPCTensor with secret that is not 1-D.

    """
    # Checks if the list of MPCTensors are of length 1
    if not all(
        ((hasattr(item, "shape")) and item.shape == torch.Size([1]))
        for item in input_list
    ):

        raise ValueError(
            "Invalid dimension. All MPCTensors should have an 1-dimensional secret."
        )

    n = len(input_list)

    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            check = input_list[j] > input_list[j + 1]
            neg_check = 1 - check
            temp = input_list[j]
            input_list[j] = neg_check * input_list[j] + check * input_list[j + 1]
            input_list[j + 1] = neg_check * input_list[j + 1] + check * temp

    if not ascending:
        return input_list[::-1]

    return input_list
