import pytest
import torch

from sympc.session import Session
from sympc.tensor import ShareTensor


@pytest.mark.parametrize("precision", [2, 3, 4])
@pytest.mark.parametrize("base", [2, 10])
def test_send_get(clients, precision, base) -> None:
    alice_client, bob_client, james_client = clients

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)

    assert x_share == x_ptr.get()


@pytest.mark.parametrize("precision", [2, 3, 4])
@pytest.mark.parametrize("base", [2, 10])
def test_add(clients, precision, base) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])

    # testing for provided session
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, session=session)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    # testing for default values of precision and base
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x)
    y_share = ShareTensor(data=y)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr + y_ptr
    assert (x_share + y_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr + y_ptr
    assert (x_share + y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr + x_ptr
    assert (x_share + x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr + y_ptr
    assert (x_share + y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr + c
    assert (x_share + c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr + c
    assert (x_share + c) == y.get()


@pytest.mark.parametrize("precision", [2, 3, 4])
@pytest.mark.parametrize("base", [2, 10])
def test_sub(clients, precision, base) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])

    # testing for provided session
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, session=session)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    # testing for default values of precision and base
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x)
    y_share = ShareTensor(data=y)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr - y_ptr
    assert (x_share - y_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr - y_ptr
    assert (x_share - y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr - x_ptr
    assert (x_share - x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr - y_ptr
    assert (x_share - y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr - c
    assert (x_share - c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr - c
    assert (x_share - c) == y.get()


@pytest.mark.parametrize("precision", [2, 3, 4])
@pytest.mark.parametrize("base", [2, 10])
def test_mul(clients, precision, base) -> None:
    alice_client, bob_client, james_client = clients
    session = Session(parties=[alice_client, bob_client, james_client])

    # testing for provided session
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, session=session)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    # testing for default values of precision and base
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x)
    y_share = ShareTensor(data=y)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr * y_ptr
    assert (x_share * y_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    x = torch.Tensor([0.122, 1.342, 4.67])
    y = torch.Tensor([1, 5.3, 4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr * y_ptr
    assert (x_share * y_share) == y.get()

    # with negative numbers
    x = torch.Tensor([0.122, -1.342, -4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y = x_ptr * x_ptr
    assert (x_share * x_share) == y.get()

    x = torch.Tensor([-0.122, 1.342, -4.67])
    y = torch.Tensor([1, -5.3, -4.678])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    y_share = ShareTensor(data=y, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    y_ptr = y_share.send(alice_client)
    y = x_ptr * y_ptr
    assert (x_share * y_share) == y.get()

    # with int constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4
    y = x_ptr * c
    assert (x_share * c) == y.get()

    # with float constant
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(data=x, encoder_precision=precision, encoder_base=base)
    x_ptr = x_share.send(alice_client)
    c = 4.6
    y = x_ptr * c
    assert (x_share * c) == y.get()
