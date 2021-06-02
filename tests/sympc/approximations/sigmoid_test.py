# third party
import pytest
import torch

from sympc.approximations.sigmoid import sigmoid
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor.mpc_tensor import MPCTensor


@pytest.mark.parametrize(
    "method", ["maclaurin", "exp", "chebyshev", "chebyshev-crypten"]
)
def test_sigmoid(get_clients, method) -> None:
    clients = get_clients(2)

    x_secret = torch.Tensor([0.0, 1, -2, 3, -4])
    x_secret_sigmoid = torch.sigmoid(x_secret)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=x_secret, session=session)
    x_sigmoid = sigmoid(x, method)

    assert torch.allclose(x_secret_sigmoid, x_sigmoid.reconstruct(), atol=1e-1)
