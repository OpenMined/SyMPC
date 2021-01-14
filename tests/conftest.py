# third party
import pytest
import syft as sy

import sympc  # noqa: 401


@pytest.fixture
def clients():
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")
    alice_client = alice.get_client()
    bob_client = bob.get_client()

    return [alice_client, bob_client]
