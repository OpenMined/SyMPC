# third party
import pytest
import torch
import torch.nn.functional as F

from sympc.approximations.softmax import log_softmax
from sympc.approximations.softmax import softmax
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize(
    "dim",
    [None, -2, -1, 0, 1],
)
def test_softmax(get_clients, dim) -> None:
    clients = get_clients(2)

    x_secret = torch.arange(-6, 6, dtype=torch.float).view(3, 4)
    x_secret_softmax = F.softmax(x_secret, dim=dim)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_softmax = softmax(x, dim=dim)

    assert torch.allclose(x_secret_softmax, x_softmax.reconstruct(), atol=1e-2)


def test_softmax_single_along_dim(get_clients) -> None:
    clients = get_clients(2)

    x_secret = torch.arange(4, dtype=torch.float).view(4, 1)
    x_secret_softmax = F.softmax(x_secret)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_softmax = softmax(x)

    assert torch.allclose(x_secret_softmax, x_softmax.reconstruct(), atol=1e-2)


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize(
    "dim",
    [None, -2, -1, 0, 1],
)
def test_log_softmax(get_clients, dim) -> None:
    clients = get_clients(2)

    x_secret = torch.arange(-6, 6, dtype=torch.float).view(3, 4)
    x_secret_log_softmax = F.log_softmax(x_secret, dim=dim)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_log_softmax = log_softmax(x, dim=dim)

    assert torch.allclose(x_secret_log_softmax, x_log_softmax.reconstruct(), atol=1e-2)


def test_log_softmax_single_along_dim(get_clients) -> None:
    clients = get_clients(2)

    x_secret = torch.arange(4, dtype=torch.float).view(4, 1)
    x_secret_log_softmax = F.log_softmax(x_secret)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_log_softmax = log_softmax(x)

    assert torch.allclose(x_secret_log_softmax, x_log_softmax.reconstruct(), atol=1e-2)
