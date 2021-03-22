# third party
import torch

from sympc.approximations.exponential import exp
from sympc.config import Config
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_exp(get_clients) -> None:
    clients = get_clients(2)

    x_secret = torch.Tensor([0.0, 1, -2, 3, -4])
    x_secret_exp = torch.exp(x_secret)

    # with custom precision
    config = Config(encoder_precision=20)
    session = Session(parties=clients, config=config)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_exp = exp(x)

    assert torch.allclose(x_secret_exp, x_exp.reconstruct(), atol=1e-1)
