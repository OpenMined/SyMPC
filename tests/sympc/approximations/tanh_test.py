# third party
import pytest
import torch

from sympc.approximations.tanh import tanh
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_tanh(get_clients) -> None:
    clients = get_clients(2)

    x_secret = torch.Tensor([0.0, 1, -2, 3, -4])
    x_secret_tanh = torch.tanh(x_secret)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_tanh = tanh(x, method="sigmoid")

    assert torch.allclose(x_secret_tanh, x_tanh.reconstruct(), atol=1e-2)

    with pytest.raises(ValueError):
        x_tanh = tanh(x, method="exp")
