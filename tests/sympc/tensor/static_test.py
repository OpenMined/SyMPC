# third party
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor
from sympc.tensor.static import cat
from sympc.tensor.static import stack


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
