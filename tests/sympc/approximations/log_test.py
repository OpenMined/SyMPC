# third party
import torch

from sympc.approximations.log import log
from sympc.config import Config
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_log(get_clients) -> None:
    clients = get_clients(2)
    x_secret = torch.Tensor([0.1, 0.5, 2, 5, 10])
    x_secret_log = torch.log(x_secret)

    # with custom precision
    config = Config(encoder_precision=20)
    session = Session(parties=clients, config=config)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=x_secret, session=session)
    x_log = log(x)

    assert torch.allclose(x_secret_log, x_log.reconstruct(), atol=1e-1)
