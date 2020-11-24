# third party
import pytest
import syft as sy

from typing import Any
from typing import List
from typing import Callable


import pytest
import syft as sy


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return [
            sy.VirtualMachine(name=f"P_{i}").get_client() for i in range(nr_clients)
        ]

    return _helper_get_clients
