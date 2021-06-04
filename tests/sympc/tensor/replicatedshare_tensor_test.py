# third party
import pytest
import torch

from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import ReplicatedSharedTensor
from sympc.utils import get_type_from_ring


def test_import_RSTensor() -> None:

    ReplicatedSharedTensor()


@pytest.mark.parametrize("precision", [12, 3, 5])
@pytest.mark.parametrize("base", [4, 6, 2, 10])
def test_fixed_point(precision, base) -> None:
    x = torch.tensor([1.25, 3.301])
    shares = [x, x]
    rst = ReplicatedSharedTensor(
        shares=shares, config=Config(encoder_precision=precision, encoder_base=base)
    )
    fp_encoder = FixedPointEncoder(precision=precision, base=base)
    tensor_type = get_type_from_ring(rst.ring_size)
    for i in range(len(shares)):
        shares[i] = fp_encoder.encode(shares[i]).to(tensor_type)

    assert (torch.cat(shares) == torch.cat(rst.shares)).all()

    for i in range(len(shares)):
        shares[i] = fp_encoder.decode(shares[i].type(torch.LongTensor))

    assert (torch.cat(shares) == torch.cat(rst.decode())).all()


@pytest.mark.skip(reason="Will be refactored to use session_uuid")
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


@pytest.mark.skip(reason="Will be refactored to use session_uuid")
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
