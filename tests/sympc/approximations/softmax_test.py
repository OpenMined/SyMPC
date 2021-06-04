# third party
import torch
import torch.nn.functional as F

from sympc.approximations.softmax import softmax
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


def test_softmax(get_clients) -> None:
    clients = get_clients(2)

    # x_secret = torch.Tensor([3])
    x_secret = torch.arange(-6, 6, dtype=torch.float)
    # x_secret = torch.arange(-6, 6, dtype=torch.float).view(2, 2, 3)
    x_secret_softmax = F.softmax(x_secret)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_softmax = softmax(x)

    assert torch.allclose(x_secret_softmax, x_softmax.reconstruct(), atol=1e-2)
