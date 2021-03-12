# third party
import torch

from sympc.approximations.exponential import exp
from sympc.config import Config
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_exp(get_clients) -> None:
    clients = get_clients(2)
    session_one = Session(parties=clients)
    SessionManager.setup_mpc(session_one)

    x_secret = torch.Tensor([0.0, 1, 2, 3, 4])

    # default precision
    x = MPCTensor(secret=x_secret, session=session_one)
    x_exp = exp(x)
    x_secret_exp = torch.exp(x_secret)

    assert torch.allclose(x_secret_exp, x_exp.reconstruct(), rtol=0.4)

    # with custom precision
    config = Config(encoder_precision=20)
    session_two = Session(parties=clients, config=config)
    SessionManager.setup_mpc(session_two)

    x = MPCTensor(secret=x_secret, session=session_two)
    x_exp = exp(x)

    assert torch.allclose(x_secret_exp, x_exp.reconstruct(), rtol=0.3)
