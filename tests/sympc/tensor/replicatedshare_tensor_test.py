# stdlib
import operator
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


@pytest.mark.parametrize("parties", [3, 5])
@pytest.mark.parametrize("security", ["malicious", "semi-honest"])
def test_rst_distribute_reconstruct_float_secret(
    get_clients, parties, security
) -> None:
    parties = get_clients(parties)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret = 43.2
    a = MPCTensor(secret=secret, session=session)
    assert np.allclose(secret, a.reconstruct(), atol=1e-3)


@pytest.mark.parametrize("parties", [3, 5])
@pytest.mark.parametrize("security", ["malicious", "semi-honest"])
def test_rst_distribute_reconstruct_tensor_secret(
    get_clients, parties, security
) -> None:
    parties = get_clients(parties)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor(
        [[1, -2.0, 0.0], [3.9, -4.394, -0.9], [-43, 100, -0.4343], [1.344, -5.0, 0.55]]
    )

    a = MPCTensor(secret=secret, session=session)
    assert np.allclose(secret, a.reconstruct(), atol=1e-3)


@pytest.mark.parametrize("security", ["malicious", "semi-honest"])
def test_rst_reconstruct_zero_share_ptrs(get_clients, security) -> None:
    parties = get_clients(3)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor(
        [[1, -2.0, 0.0], [3.9, -4.394, -0.9], [-43, 100, -0.4343], [1.344, -5.0, 0.55]]
    )

    a = MPCTensor(secret=secret, session=session)
    a.share_ptrs = []
    with pytest.raises(ValueError):
        a.reconstruct()


@pytest.mark.parametrize("parties", [2, 5])
def test_share_distribution_number_shares(get_clients, parties):
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

    secret = torch.Tensor(
        [[1, -2.0, 0.0], [3.9, -4.394, -0.9], [-43, 100, -0.4343], [1.344, -5.0, 0.55]]
    )

    tensor = MPCTensor(secret=secret, session=session)
    tensor.share_ptrs[0][0] = tensor.share_ptrs[0][0] + 4

    with pytest.raises(ValueError):
        tensor.reconstruct()


@pytest.mark.parametrize("op_str", ["add", "sub"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ops_share_private(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ReplicatedSharedTensor(
        shares=[x], config=Config(encoder_base=base, encoder_precision=precision)
    )
    y_share = ReplicatedSharedTensor(
        shares=[y], config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y_share)
    tensor_decoded = res.fp_encoder.decode(res.shares[0])

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


@pytest.mark.parametrize("op_str", ["add", "sub"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ops_share_public(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ReplicatedSharedTensor(
        shares=[x], config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y)
    tensor_decoded = res.fp_encoder.decode(res.shares[0])

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


def test_rst_resolve_pointer(get_clients) -> None:
    clients = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=clients)
    SessionManager.setup_mpc(session)
    secret = torch.randn(1, 2)
    tensor = MPCTensor(secret=secret, session=session)

    share_pt0 = tensor.share_ptrs[0]
    resolved_share_pt0 = share_pt0.resolve_pointer_type()
    share_pt_name = type(resolved_share_pt0).__name__

    assert share_pt_name == "ReplicatedSharedTensorPointer"


@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
@pytest.mark.parametrize("security", ["semi-honest"])  # malicious to be added
def test_ops_public_mul(get_clients, security, base, precision):
    parties = get_clients(3)
    protocol = Falcon(security)
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([[0.125, 1001, -1.25, 4.82], [-4.25, 0.217, 3301, 4]])
    value = 8

    tensor = MPCTensor(secret=secret, session=session)
    result = tensor * value
    expected_res = secret * value

    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
@pytest.mark.parametrize("security", ["semi-honest"])  # malicous to be addded
def test_ops_public_mul_matrix(get_clients, security, base, precision):
    parties = get_clients(3)
    protocol = Falcon(security)
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([[0.125, 1001, 4.82, -1.25], [-4.25, 0.217, 3301, 4]])
    value = torch.Tensor([[4.5, 9.25, 3.47, -2.5], [50, 3.17, 5.82, 2.25]])

    tensor = MPCTensor(secret=secret, session=session)
    result = tensor * value
    expected_res = secret * value
    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


@pytest.mark.parametrize("parties", [2, 3, 5])
@pytest.mark.parametrize("security", ["semi-honest"])
def test_ops_public_mul_integer_parties(get_clients, parties, security):

    config = Config(encoder_base=1, encoder_precision=0)

    parties = get_clients(parties)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    value = 8

    op = getattr(operator, "mul")
    tensor = MPCTensor(secret=secret, session=session)
    shares = [op(share, value) for share in tensor.share_ptrs]
    result = MPCTensor(shares=shares, session=session)

    assert (result.reconstruct() == (secret * value)).all()


def test_truediv_exception() -> None:
    rst = ReplicatedSharedTensor(shares=[1])
    with pytest.raises(ValueError):
        rst / 1.55


def test_truediv() -> None:
    secret = 10.25
    rst = ReplicatedSharedTensor(shares=[secret])
    rst = rst / 2
    expected_res = rst.fp_encoder.encode(secret) // 2
    assert rst.shares[0] == expected_res


def test_rshift_exception() -> None:
    rst = ReplicatedSharedTensor(shares=[1])
    with pytest.raises(ValueError):
        rst >> 1.55


def test_rshift() -> None:
    secret = 10.25
    rst = ReplicatedSharedTensor(shares=[secret])
    rst = rst >> 2
    expected_res = rst.fp_encoder.encode(secret) >> 2
    assert rst.shares[0] == expected_res
