# third party
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import ReplicatedSharedTensor


def test_import_RSTensor() -> None:

    ReplicatedSharedTensor()


def test_hook_method(get_clients) -> None:
    alice, bob, charlie = get_clients(3)
    session = Session(parties=[alice, bob, charlie])
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)  # noqa: F841
    shares = [x, y]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert rst.numel() == x.numel()
    assert (rst.t().shares[0] == x.t()).all()
    assert (rst.unsqueeze(dim=0).shares[0] == x.unsqueeze(dim=0)).all()
    assert (rst.view(3, 1).shares[0] == x.view(3, 1)).all()
    assert (rst.sum().shares[0] == x.sum()).all()


def test_hook_property(get_clients) -> None:
    alice, bob, charlie = get_clients(3)
    session = Session(parties=[alice, bob, charlie])
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)  # noqa: F841
    shares = [x, y]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert (rst.T.shares[0] == x.T).all()
