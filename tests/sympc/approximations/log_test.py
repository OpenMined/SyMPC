# third party
import torch

from sympc.approximations.log import log
from sympc.config import Config
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_log(get_clients) -> None:
    clients = get_clients(2)
    session_one = Session(parties=clients)
    SessionManager.setup_mpc(session_one)

    x_secret = torch.Tensor([0.1, 0.5, 2, 5, 10, 20, 50, 100, 250])

    # default precision
    x = MPCTensor(secret=x_secret, session=session_one)
    x_log = log(x)
    x_secret_log = torch.log(x_secret)

    assert torch.allclose(x_secret_log, x_log.reconstruct(), rtol=0.4)

    # with custom precision
    config = Config(encoder_precision=20)
    session_two = Session(parties=clients, config=config)
    SessionManager.setup_mpc(session_two)

    x = MPCTensor(secret=x_secret, session=session_two)
    x_log = log(x)

    assert torch.allclose(x_secret_log, x_log.reconstruct(), rtol=0.3)
