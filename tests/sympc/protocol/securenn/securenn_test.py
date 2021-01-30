import torch

from sympc.session import Session
from sympc.tensor import MPCTensor


def test_private_compare(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    Session.setup_mpc(session)

    x_secret = torch.Tensor([1, -2])
    x = MPCTensor(secret=x_secret, session=session)

    print(x > torch.tensor(5))
