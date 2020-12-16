import pytest
import syft as sy


@pytest.fixture
def clients():
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")

    return [alice_client, bob_client]
