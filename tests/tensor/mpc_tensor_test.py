import torch

from sympc.session import Session
from sympc.tensor import MPCTensor


def test_matmul(clients):
    alice_client, bob_client = clients
    # TODO: for more than 2 parties
    session = Session(parties=[alice_client, bob_client])
    Session.setup_mpc(session)

    x_secret = torch.Tensor([[1, 22], [3, 41], [-7, 4]])
    y_secret = torch.Tensor([[28], [14]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x @ y).reconstruct()
    result_secret = x_secret @ y_secret
    assert torch.allclose(result, result_secret, atol=10e-3)

    # With floats
    x_secret = torch.Tensor([[1.1, 2.3], [3, 41], [-7, 4]])
    y_secret = torch.Tensor([[2.8], [-1.4]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = (x @ y).reconstruct()
    result_secret = x_secret @ y_secret
    assert torch.allclose(result, result_secret, atol=10e-3)
