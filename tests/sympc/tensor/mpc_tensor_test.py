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


def test_reconstruct(clients) -> None:
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, -2, 3.0907, -4.870])
    x = MPCTensor(secret=x_secret, session=session)
    x = x.reconstruct()

    assert torch.allclose(x_secret, x)


def test_sessions_without_setup_mpc(clients):
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])

    with pytest.raises(ValueError):
        x = MPCTensor(secret=torch.Tensor([1, -2]), session=session)


def test_sessions_with_different_sessions(clients):
    alice_client, bob_client = clients
    session_one = Session(parties=[alice_client, bob_client])
    session_two = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session_one)
    Session.setup_mpc(session_two)

    x = MPCTensor(secret=torch.Tensor([1, -2]), session=session_one)
    y = MPCTensor(secret=torch.Tensor([1, -2]), session=session_two)

    with pytest.raises(ValueError):
        z = x + y


def test_remote_mpc_no_shape(clients):
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_remote = alice_client.torch.Tensor([1, -2, 0.3])

    with pytest.raises(ValueError):
        x = MPCTensor(secret=x_remote, session=session)


def test_remote_mpc_with_shape(clients):
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_remote = alice_client.torch.Tensor([1, -2, 0.3])
    x = MPCTensor(secret=x_remote, shape=(1, 3), session=session)
    result = x.reconstruct()

    assert x_remote == result


def test_remote_not_tensor(clients):
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_remote_int = bob_client.python.Int(5)
    x = MPCTensor(secret=x_remote_int, shape=(1,), session=session)
    result = x.reconstruct()

    assert x_remote_int == result

    x_remote_int = bob_client.python.Float(5.4)
    x = MPCTensor(secret=x_remote_int, shape=(1,), session=session)
    result = x.reconstruct()

    assert x_remote_int == result


def test_local_secret_not_tensor(clients):
    # TODO: float
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_int = 5
    x = MPCTensor(secret=x_int, session=session)
    result = x.reconstruct()

    assert x_int == result


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
def test_ops(clients, op_str) -> None:
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


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "truediv"])
def test_ops_mpc_public(clients, op_str) -> None:
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    # TODO: support for matmul
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)
    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y_secret)

    if op_str == "truediv":
        with pytest.raises(ValueError):
            result = op(x, y_secret).reconstruct()
    else:
        result = op(x, y_secret).reconstruct()
        assert torch.allclose(result, expected_result, atol=10 ** -3)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_ops_public_mpc(clients, op_str) -> None:
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    # TODO: support for matmul
    # TODO: support for div
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(y_secret, x_secret)
    result = op(y_secret, x).reconstruct()

    assert torch.allclose(result, expected_result, atol=10 ** -3)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
def test_ops_integer(clients, op_str) -> None:
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    # TODO: support for matmul
    # TODO: support for div
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([0.125, -1.25, -4.25, 4])
    y = 4

    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y)
    result = op(x, y).reconstruct()

    assert torch.allclose(result, expected_result, atol=10 ** -3)


def test_mpc_print(clients) -> None:
    alice_client, bob_client = clients
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([5.0])

    x = MPCTensor(secret=x_secret, session=session)

    expected = "[MPCTensor]\n\t|"
    expected = f"{expected} {alice_client} -> ShareTensorPointer\n\t|"
    expected = f"{expected} {bob_client} -> ShareTensorPointer"

    assert expected == x.__str__()