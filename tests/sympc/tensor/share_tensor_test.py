import pytest
import syft as sy
import torch

from sympc.session import Session
from sympc.tensor import ShareTensor


@pytest.fixture
def clients():
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")
    james = sy.VirtualMachine(name="james")

    alice_client = alice.get_client()
    bob_client = bob.get_client()
    james_client = james.get_client()

    return [alice_client, bob_client, james_client]


def test_send_get(clients) -> None:
    alice_client, bob_client, james_client = clients

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)

    assert x_share == x_ptr.get()


def test_add(clients) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])
    Session.setup_mpc(session)

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr + y_ptr
    assert (x_share + y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr + y_ptr
    assert (x_share + y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr + c
    assert (x_share + c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr + c
    assert (x_share + c) == y.get()


def test_sub(clients) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])
    Session.setup_mpc(session)

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr - y_ptr
    assert (x_share - y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr - y_ptr
    assert (x_share - y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr - c
    assert (x_share - c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr - c
    assert (x_share - c) == y.get()


def test_mul(clients) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])
    Session.setup_mpc(session)

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr * y_ptr
    assert (x_share * y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    y_share = ShareTensor(data=y, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr * y_ptr
    assert (x_share * y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr * c
    assert (x_share * c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=4, encoder_base=10)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr * c
    assert (x_share * c) == y.get()
