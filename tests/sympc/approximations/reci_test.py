# third party
import torch

from sympc.approximations.reci import reciprocal
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_reciprocal(get_clients) -> None:
    clients = get_clients(2)
    session_one = Session(parties=clients)
    SessionManager.setup_mpc(session_one)

    x_secret = torch.Tensor([-2.0, 6.0, 2.0, 3.0, -5.0, -0.5])

    x = MPCTensor(secret=x_secret, session=session_one)
    x_secret_reciprocal = torch.reciprocal(x_secret)

    x_reciprocal = reciprocal(x, method="nr")
    assert torch.allclose(x_secret_reciprocal, x_reciprocal.reconstruct(), atol=1e-2)

    x_reciprocal = reciprocal(x, method="log")
    assert torch.allclose(x_secret_reciprocal, x_reciprocal.reconstruct(), atol=1e-2)
