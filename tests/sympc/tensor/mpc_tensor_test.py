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


def test_add(clients):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, 2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert torch.allclose(result, (x_secret + x_secret))

    x_secret = torch.Tensor([1, 2, 3])
    y_secret = torch.Tensor([4, 5, 6])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert torch.allclose(result, (x_secret + y_secret))

    # with negative numbers
    x_secret = torch.Tensor([1, -2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert torch.allclose(result, (x_secret + x_secret))

    x_secret = torch.Tensor([1, -2, 3])
    y_secret = torch.Tensor([4, 5, -6])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert torch.allclose(result, (x_secret + y_secret))

    # with constant integer
    x_secret = torch.Tensor([1, 2, 3])
    c = 4
    x = MPCTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    assert torch.allclose(result, (x_secret + c))

    # with constant float
    x_secret = torch.Tensor([1, 2, 3])
    c = 4.6
    x = MPCTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    result_c = x_secret + c
    assert torch.allclose(result, result_c)

    # test with different sessions
    session_alternate = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session_alternate)

    x_secret = torch.Tensor([1, 2, 3])

    x_session = MPCTensor(secret=x_secret, session=session)
    x_session_alternate = MPCTensor(secret=x_secret, session=session_alternate)

    with pytest.raises(ValueError):
        result = x_session + x_session_alternate


def test_sub(clients):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, 2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x - x).reconstruct()
    assert torch.allclose(result, (x_secret - x_secret))

    x_secret = torch.Tensor([1, 2, 3])
    y_secret = torch.Tensor([4, 5, 6])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x - y).reconstruct()
    assert torch.allclose(result, (x_secret - y_secret))

    # with negative numbers
    x_secret = torch.Tensor([1, -2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x - x).reconstruct()
    assert torch.allclose(result, (x_secret - x_secret))

    x_secret = torch.Tensor([1, -2, 3])
    y_secret = torch.Tensor([4, 5, -6])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x - y).reconstruct()
    assert torch.allclose(result, (x_secret - y_secret))

    # with constant integer
    x_secret = torch.Tensor([1, 2, 3])
    c = 4
    x = MPCTensor(secret=x_secret, session=session)
    result = (x - c).reconstruct()
    assert torch.allclose(result, (x_secret - c))

    # with constant float
    x_secret = torch.Tensor([1, 2, 3])
    c = 4.6
    x = MPCTensor(secret=x_secret, session=session)
    result = (x - c).reconstruct()
    result_c = x_secret - c
    assert torch.allclose(result, result_c)

    # test with different sessions
    session_alternate = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session_alternate)

    x_secret = torch.Tensor([1, 2, 3])

    x_session = MPCTensor(secret=x_secret, session=session)
    x_session_alternate = MPCTensor(secret=x_secret, session=session_alternate)

    with pytest.raises(ValueError):
        result = x_session - x_session_alternate


def test_mul(clients):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, 2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x * x).reconstruct()
    result_secret = x_secret * x_secret
    assert torch.allclose(result, result_secret)

    x_secret = torch.Tensor([1, 2, 3])
    y_secret = torch.Tensor([4, 5, 6])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x * y).reconstruct()
    result_secret = x_secret * y_secret
    assert torch.allclose(result, result_secret)

    # with negative numbers
    x_secret = torch.Tensor([1, -2, 3])
    x = MPCTensor(secret=x_secret, session=session)
    result = (x * x).reconstruct()
    result_secret = x_secret * x_secret
    assert torch.allclose(result, result_secret)

    # with constant integer
    x_secret = torch.Tensor([1, 2, 3])
    c = 4
    x = MPCTensor(secret=x_secret, session=session)
    result = (x * c).reconstruct()
    result_secret = x_secret * c
    assert torch.allclose(result, result_secret)

    # with constant float
    x_secret = torch.Tensor([1, 2, 3])
    c = 4.6
    x = MPCTensor(secret=x_secret, session=session)
    result = (x * c).reconstruct()
    result_secret = x_secret * c
    assert torch.allclose(result, result_secret)

    # test with different sessions
    session_alternate = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session_alternate)

    x_secret = torch.Tensor([1, 2, 3])

    x_session = MPCTensor(secret=x_secret, session=session)
    x_session_alternate = MPCTensor(secret=x_secret, session=session_alternate)

    with pytest.raises(ValueError):
        result = x_session * x_session_alternate
