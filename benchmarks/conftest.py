"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import pytest
import syft as sy


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    """Generate a list of clients given a number.

    Returns:
        Callable[[int], List[Any]]: List of clients
    """

    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return [
            sy.VirtualMachine(name=f"P_{i}").get_root_client()
            for i in range(nr_clients)
        ]

    return _helper_get_clients
