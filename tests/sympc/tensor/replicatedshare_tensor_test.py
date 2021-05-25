# third party
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor import ShareTensor


def test_import_RSTensor() -> None:

    ReplicatedSharedTensor()


def test_hook_method(get_clients) -> None:
    alice, bob = get_clients(2)
    session = Session(parties=[alice, bob])
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)  # noqa: F841
    stx = ShareTensor(data=x, session=session)
    sty = ShareTensor(data=y, session=session)
    shares = [stx, sty]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert rst.numel() == stx.numel()
    assert rst.t().shares[0] == stx.t()
    assert rst.unsqueeze(dim=0).shares[0] == stx.unsqueeze(dim=0)
    assert rst.view(3, 1).shares[0] == stx.view(3, 1)
    assert rst.sum().shares[0] == stx.sum()


def test_hook_property(get_clients) -> None:
    alice, bob = get_clients(2)
    session = Session(parties=[alice, bob])
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)  # noqa: F841
    stx = ShareTensor(data=x, session=session)
    sty = ShareTensor(data=y, session=session)
    shares = [stx, sty]

    rst = ReplicatedSharedTensor(shares=shares, session=session)

    assert rst.T.shares[0] == stx.T
