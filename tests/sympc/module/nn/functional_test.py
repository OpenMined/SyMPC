# third party
import numpy as np
import torch

from sympc.module.nn import mse_loss
from sympc.module.nn import relu
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


def test_relu(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([-2, -1.5, 0, 1, 1.5, 2])
    mpc_tensor = MPCTensor(secret=secret, session=session)

    res = relu(mpc_tensor)
    res_expected = torch.nn.functional.relu(secret)

    assert all(res.reconstruct() == res_expected)


def test_mse_loss(get_clients) -> None:
    clients = get_clients(4)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    y_secret = torch.Tensor([0.23, 0.32, 0.2, 0.3])
    y_mpc = MPCTensor(secret=y_secret, session=session)

    y_pred = torch.Tensor([0.1, 0.3, 0.4, 0.2])
    y_pred_mpc = MPCTensor(secret=y_pred, session=session)

    res = mse_loss(y_mpc, y_pred_mpc)
    res_expected = torch.nn.functional.mse_loss(y_secret, y_pred, reduction="sum")

    assert np.allclose(res.reconstruct(), res_expected, atol=1e-4)
