# stdlib
import itertools

# third party
import pytest
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor
from sympc.tensor.static import cat
from sympc.tensor.static import stack


def test_argmax_multiple_max(get_clients) -> None:
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

    expected = secret.argmax().float()
    res = argmax_val.reconstruct()
    assert torch.equal(res, expected), f"Expected argmax to be {expected}"


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
    expected = secret.argmax(dim=dim, keepdim=keepdim).float()
    assert torch.equal(res, expected), f"Expected argmax to be {expected}"


def test_max_multiple_max(get_clients) -> None:
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

    expected = secret.max().float()
    res = max_val.reconstruct()
    assert torch.equal(res, expected), f"Expected argmax to be {expected}"


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
    expected = secret.max(dim=dim, keepdim=keepdim).values
    assert torch.equal(res, expected), f"Expected argmax to be {expected}"


def test_stack(get_clients):
    clients = get_clients(2)

    x_secret = torch.Tensor([0.0, 1, -2, 3, -4])
    y_secret = torch.Tensor([-4, 3, -2, 1, 0.0])
    secret_stacked = torch.stack([x_secret, y_secret])

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    stacked = stack([x, y])

    assert torch.equal(secret_stacked, stacked.reconstruct())


def test_cat(get_clients):
    clients = get_clients(2)

    x_secret = torch.Tensor([0.0, 1, -2, 3, -4])
    y_secret = torch.Tensor([-4, 3, -2, 1, 0.0])
    secret_concatenated = torch.cat([x_secret, y_secret])

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    concatenated = cat([x, y])

    assert torch.equal(secret_concatenated, concatenated.reconstruct())
