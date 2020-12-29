# stdlib
import operator

# third party
import pytest
import torch

from sympc.session import Session
from sympc.tensor import MPCTensor


def test_mpc_tensor_exception(clients) -> None:
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])

    with pytest.raises(ValueError):
        MPCTensor(secret=42, session=session)


def test_reconstruct(clients):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, 2, 3, 4])
    x = MPCTensor(secret=x_secret, session=session)
    x = x.reconstruct()

    assert torch.allclose(x_secret, x)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
def test_ops(clients, op_str):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert torch.allclose(result, expected_result)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_ops_mpc_secret(clients, op_str):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y_secret)
    result = op(x, y_secret).reconstruct()

    assert torch.allclose(result, expected_result)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_ops_secret_mpc(clients, op_str):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(y_secret, x_secret)
    result = op(y_secret, x).reconstruct()

    assert torch.allclose(result, expected_result)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_ops_integer(clients, op_str):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([0.125, -1.25, -4.25, 4])
    y = 4

    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y)
    result = op(x, y).reconstruct()

    assert torch.allclose(result, expected_result)
