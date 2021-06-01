# stdlib
import itertools

# third party
import pytest
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor


def test_argmax_mutiple_max(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=torch.Tensor([1, 2, 3, -1, 3]), session=session)

    with pytest.raises(ValueError):
        x.argmax()


def test_argmax(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([1, 2, 3, -1, -3])
    x = MPCTensor(secret=secret, session=session)

    argmax_val = x.argmax()
    assert isinstance(x, MPCTensor), "Expected argmax to be MPCTensor"

    expected = secret.argmax()
    res = argmax_val.reconstruct()
    assert res == expected, f"Expected argmax to be {expected}"


@pytest.mark.parametrize("dim, keepdim", itertools.product([0, 1, 2], [True, False]))
def test_argmax_dim(dim, keepdim, get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([[[1, 2], [3, -1], [4, 5]], [[2, 5], [5, 1], [6, 42]]])
    x = MPCTensor(secret=secret, session=session)

    argmax_val = x.argmax(dim=dim, keepdim=keepdim)
    assert isinstance(x, MPCTensor), "Expected argmax to be MPCTensor"

    res = argmax_val.reconstruct()
    expected = secret.argmax(dim=dim, keepdim=keepdim)
    assert (res == expected).all(), f"Expected argmax to be {expected}"


def test_max_mutiple_max(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=torch.Tensor([1, 2, 3, -1, 3]), session=session)

    with pytest.raises(ValueError):
        x.argmax()


def test_max(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([1, 2, 3, -1, -3])
    x = MPCTensor(secret=secret, session=session)

    max_val = x.max()
    assert isinstance(x, MPCTensor), "Expected argmax to be MPCTensor"

    expected = secret.max()
    res = max_val.reconstruct()
    assert res == expected, f"Expected argmax to be {expected}"


@pytest.mark.parametrize("dim, keepdim", itertools.product([0, 1, 2], [True, False]))
def test_max_dim(dim, keepdim, get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([[[1, 2], [3, -1], [4, 5]], [[2, 5], [5, 1], [6, 42]]])
    x = MPCTensor(secret=secret, session=session)

    max_val = x.max(dim=dim, keepdim=keepdim)
    assert isinstance(x, MPCTensor), "Expected argmax to be MPCTensor"

    res = max_val.reconstruct()
    expected = secret.max(dim=dim, keepdim=keepdim)
    assert (res == expected).all(), f"Expected argmax to be {expected}"
