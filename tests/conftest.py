"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import pytest
import syft as sy

vms = [sy.VirtualMachine(name=f"P_{i}") for i in range(11)]


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    def _helper_get_clients(nr_clients: int) -> List[Any]:
        shared_vms = [vm for vm in vms[0:nr_clients]]
        clients = [vm.get_root_client() for vm in shared_vms]
        return clients

    return _helper_get_clients
