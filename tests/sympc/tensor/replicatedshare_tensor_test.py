# stdlib
from uuid import uuid4

# third party
import numpy as np
import pytest
import torch

from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor
from sympc.utils import get_type_from_ring


def test_import_RSTensor() -> None:
    ReplicatedSharedTensor()


def test_different_session_ids() -> None:
    x = torch.randn(1)
    shares = [x, x]
    x_share = ReplicatedSharedTensor(shares=shares, session_uuid=uuid4())
    y_share = ReplicatedSharedTensor(shares=shares, session_uuid=uuid4())

    # Different session ids
    assert x_share != y_share


def test_same_session_id_and_data() -> None:
    x = torch.randn(1)
    shares1 = [x, x]
    y = torch.randn(1)
    shares2 = [y, y]
    session_id = uuid4()
    x_share = ReplicatedSharedTensor(shares=shares1, session_uuid=session_id)
    y_share = ReplicatedSharedTensor(shares=shares2, session_uuid=session_id)

    # Different shares list
    assert x_share != y_share


def test_different_config() -> None:
    x = torch.randn(1)
    shares = [x, x]
    session_id = uuid4()
    config1 = Config(encoder_precision=10, encoder_base=2)
    config2 = Config(encoder_precision=12, encoder_base=10)
    x_share = ReplicatedSharedTensor(
        shares=shares, session_uuid=session_id, config=config1
    )
    y_share = ReplicatedSharedTensor(
        shares=shares, session_uuid=session_id, config=config2
    )

    # Different fixed point config
    assert x_share != y_share


def test_send_get(get_clients, precision=12, base=4) -> None:
    client = get_clients(1)[0]
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=[client])
    SessionManager.setup_mpc(session)
    share1 = torch.Tensor([1.4, 2.34, 3.43])
    share2 = torch.Tensor([1, 2, 3])
    share3 = torch.Tensor([1.4, 2.34, 3.43])
    session_uuid = session.rank_to_uuid[0]
    x_share = ReplicatedSharedTensor(
        shares=[share1, share2, share3], session_uuid=session_uuid
    )
    x_ptr = x_share.send(client)
    result = x_ptr.get()

    assert result == x_share


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


def test_hook_method(get_clients) -> None:
    clients = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=clients)
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)
    shares = [x, y]

    rst = ReplicatedSharedTensor()
    rst.shares = shares

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
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=clients)
    SessionManager.setup_mpc(session)

    x = torch.randn(1, 3)
    y = torch.randn(1, 3)
    shares = [x, y]

    rst = ReplicatedSharedTensor()
    rst.shares = shares

    assert (rst.T.shares[0] == x.T).all()
    assert (rst.T.shares[1] == y.T).all()


@pytest.mark.parametrize("parties", [3, 5, 11])
@pytest.mark.parametrize("security", ["malicious", "semi-honest"])
def test_rst_distribute_reconstruct(get_clients, parties, security) -> None:
    parties = get_clients(parties)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret = 42.32

    a = MPCTensor(secret=secret, session=session)

    assert np.allclose(secret, a.reconstruct(), atol=1e-5)


@pytest.mark.parametrize("parties", [2, 5, 11])
def test_share_distribution(get_clients, parties):
    parties = get_clients(parties)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    shares = MPCTensor.generate_shares(100.42, len(parties))
    share_ptrs = ReplicatedSharedTensor.distribute_shares(shares, session)

    for RSTensor in share_ptrs:
        assert len(RSTensor.get_shares().get()) == (len(parties) - 1)


@pytest.mark.parametrize("parties", [3, 5])
def test_invalid_malicious_reconstruction(get_clients, parties):
    parties = get_clients(parties)
    protocol = Falcon("malicious")
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret = 42.32

    tensor = MPCTensor(secret=secret, session=session)
    tensor.share_ptrs[0][0] = tensor.share_ptrs[0][0] + 4

    with pytest.raises(ValueError):
        tensor.reconstruct()
