import pytest
import syft as sy
import torch

from syft.lib.sympc.session import SySession
from sympc.tensor.additive_shared import AdditiveSharingTensor


@pytest.fixture
def clients():
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")
    james = sy.VirtualMachine(name="james")

    alice_client = alice.get_client()
    bob_client = bob.get_client()
    james_client = james.get_client()

    return [alice_client, bob_client, james_client]


def test_reconstruct(clients):
    alice_client, bob_client, james_client = clients
    session = SySession(parties=[alice_client, bob_client, james_client])
    session.setup_mpc()

    x_secret = torch.Tensor([1, 2, 3, 4])
    x = AdditiveSharingTensor(secret=x_secret, session=session)

    x = x.reconstruct()

    assert torch.allclose(x_secret, x)


def test_add(clients):
    alice_client, bob_client, james_client = clients
    session = SySession(parties=[alice_client, bob_client, james_client])
    session.setup_mpc()

    x_secret = torch.Tensor([1, 2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert torch.allclose(result, (x_secret + x_secret))

    x_secret = torch.Tensor([1, 2, 3])
    y_secret = torch.Tensor([4, 5, 6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert torch.allclose(result, (x_secret + y_secret))

    # with negative numbers
    x_secret = torch.Tensor([1, -2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert torch.allclose(result, (x_secret + x_secret))

    x_secret = torch.Tensor([1, -2, 3])
    y_secret = torch.Tensor([4, 5, -6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert torch.allclose(result, (x_secret + y_secret))

    # with constant integer
    x_secret = torch.Tensor([1, 2, 3])
    c = 4
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    assert torch.allclose(result, (x_secret + c))

    # with constant float
    x_secret = torch.Tensor([1, 2, 3])
    c = 4.6
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    assert torch.allclose(result, (x_secret + c))

    # test with different sessions
    session_alternate = sy.lib.sympc.session.SySession(parties=[alice_client, bob_client])
    session_alternate.setup_mpc()

    x_secret = torch.Tensor([1, 2, 3])

    x_session = AdditiveSharingTensor(secret=x_secret, session=session)
    x_session_alternate = AdditiveSharingTensor(secret=x_secret, session=session_alternate)

    with pytest.raises(ValueError):
        result = x_session + x_session_alternate


def test_sub(clients):
    alice_client, bob_client, james_client = clients
    session = SySession(parties=[alice_client, bob_client, james_client])
    session.setup_mpc()

    x_secret = torch.Tensor([1, 2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x - x).reconstruct()
    assert torch.allclose(result, (x_secret - x_secret))

    x_secret = torch.Tensor([1, 2, 3])
    y_secret = torch.Tensor([4, 5, 6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x - y).reconstruct()
    assert torch.allclose(result, (x_secret - y_secret))

    # with negative numbers
    x_secret = torch.Tensor([1, -2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x - x).reconstruct()
    assert torch.allclose(result, (x_secret - x_secret))

    x_secret = torch.Tensor([1, -2, 3])
    y_secret = torch.Tensor([4, 5, -6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x - y).reconstruct()
    assert torch.allclose(result, (x_secret - y_secret))

    # with constant integer
    x_secret = torch.Tensor([1, 2, 3])
    c = 4
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x - c).reconstruct()
    assert torch.allclose(result, (x_secret - c))

    # with constant float
    x_secret = torch.Tensor([1, 2.0, 3])
    c = 4.6
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x - c).reconstruct()
    assert torch.allclose(result, (x_secret - c))

    # test with different sessions
    session_alternate = sy.lib.sympc.session.SySession(parties=[alice_client, bob_client])
    session_alternate.setup_mpc()

    x_secret = torch.Tensor([1, 2, 3])

    x_session = AdditiveSharingTensor(secret=x_secret, session=session)
    x_session_alternate = AdditiveSharingTensor(secret=x_secret, session=session_alternate)

    with pytest.raises(ValueError):
        result = x_session - x_session_alternate
