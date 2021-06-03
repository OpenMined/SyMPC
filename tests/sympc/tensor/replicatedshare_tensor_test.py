# third party
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import ReplicatedSharedTensor


def test_import_RSTensor() -> None:

    ReplicatedSharedTensor()
    
def test_send_get(get_clients, precision=12, base=4) -> None:

    client = get_clients(1)[0]
    session = Session(parties=[client])
    SessionManager.setup_mpc(session)
    share1 = torch.Tensor([1.4, 2.34, 3.43])
    share2 = torch.Tensor([1, 2, 3])
    share3 = torch.Tensor([1.4, 2.34, 3.43])
    x_share = ReplicatedSharedTensor(shares=[share1, share2, share3], session=session)
    x_ptr = x_share.send(client)
    result = x_ptr.get()

    assert result == x_share


def test_hook_method(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)
    shares = [x, y]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert rst.numel() == x.numel()
    assert (rst.t().shares[0] == x.t()).all()
    assert (rst.unsqueeze(dim=0).shares[0] == x.unsqueeze(dim=0)).all()
    assert (rst.view(3, 1).shares[0] == x.view(3, 1)).all()
    assert (rst.sum().shares[0] == x.sum()).all()

    assert rst.numel() == y.numel()
    assert (rst.t().shares[1] == y.t()).all()
    assert (rst.unsqueeze(dim=0).shares[1] == y.unsqueeze(dim=0)).all()
    assert (rst.view(3, 1).shares[1] == y.view(3, 1)).all()
    assert (rst.sum().shares[1] == y.sum()).all()


def test_hook_property(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)
    shares = [x, y]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert (rst.T.shares[0] == x.T).all()
    assert (rst.T.shares[1] == y.T).all()
