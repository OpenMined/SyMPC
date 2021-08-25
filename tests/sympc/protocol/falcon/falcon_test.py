# stdlib
# stdlib
import operator

# third party
import numpy as np
import pytest
import torch

from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.protocol import ABY3
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.store import CryptoPrimitiveProvider
from sympc.tensor import MPCTensor
from sympc.tensor import PRIME_NUMBER
from sympc.tensor import ReplicatedSharedTensor
from sympc.utils import get_type_from_ring


def test_share_class() -> None:
    assert Falcon.share_class == ReplicatedSharedTensor


def test_session() -> None:
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol)
    assert type(session.protocol) == Falcon


def test_exception_malicious_less_parties(get_clients, parties=2) -> None:
    parties = get_clients(parties)
    protocol = Falcon("malicious")
    with pytest.raises(ValueError):
        Session(protocol=protocol, parties=parties)


def test_invalid_security_type():
    with pytest.raises(ValueError):
        Falcon(security_type="covert")


def test_eq():
    falcon = Falcon()
    aby1 = ABY3(security_type="malicious")
    aby2 = ABY3()
    other2 = falcon

    # Test equal protocol:
    assert falcon == other2

    # Test different protocol security type
    assert falcon != aby1

    # Test different protocol objects
    assert falcon != aby2


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_mul_private(get_clients, security, base, precision):
    parties = get_clients(3)
    protocol = Falcon(security)
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor(
        [[-100.25, 0.29, 30.45], [-90.82, 1000, 0.18], [1032.45, -323.18, 15.15]]
    )
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    result = tensor1 * tensor2
    expected_res = secret1 * secret2
    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_mul_private_matrix(get_clients, security, base, precision):
    parties = get_clients(3)
    protocol = Falcon(security)
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor(
        [[-100.25, 20.3, 30.12], [-50.1, 100.217, 1.2], [1032.15, -323.56, 15.15]]
    )

    secret2 = torch.tensor([[-1, 0.28, 3], [-9, 10.18, 1], [32, -23, 5]])

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    result = tensor1 * tensor2
    expected_res = secret1 * secret2
    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


@pytest.mark.parametrize("parties", [2, 4])
def test_mul_private_exception_nothreeparties(get_clients, parties):
    parties = get_clients(parties)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    with pytest.raises(ValueError):
        tensor1 * tensor2


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
def test_private_matmul(get_clients, security):
    parties = get_clients(3)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)
    secret1 = torch.tensor(
        [
            [[-100.25, 20.3, 30.12], [-50.1, 100.217, 1.2], [1032.15, -323.56, 15.15]],
            [[-0.25, 2.3, 3012], [-5.01, 1.00217, 1.2], [2.15, -3.56, 15.15]],
        ]
    )
    secret2 = torch.tensor([[-1, 0.28, 3], [-9, 10.18, 1], [32, -23, 5]])
    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)
    result = tensor1 @ tensor2
    expected_res = secret1 @ secret2
    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


def test_exception_mul_malicious(get_clients):
    parties = get_clients(3)
    protocol = Falcon("malicious")
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=1, session=session)
    y = MPCTensor(secret=2, session=session)
    shape_x = tuple(x.shape)
    shape_y = tuple(y.shape)
    p_kwargs = {"a_shape": shape_x, "b_shape": shape_y}
    tensor = torch.tensor([1])
    primitives = CryptoPrimitiveProvider.generate_primitives(
        "beaver_mul",
        session=session,
        g_kwargs={
            "session": session,
            "a_shape": shape_x,
            "b_shape": shape_y,
            "nr_parties": session.nr_parties,
        },
        p_kwargs=p_kwargs,
    )

    party = [0, 2]  # modify the primitives of party 0,2

    sess_list = [session.session_ptrs[i].get_copy() for i in party]

    for i, p in enumerate(party):
        idx = 0 if p == 0 else 1
        primitives[p][0][0].shares[idx] = tensor
        sess_list[i].crypto_store.store = {}
        sess_list[i].crypto_store.populate_store(
            "beaver_mul", primitives[p], **p_kwargs
        )
        session.session_ptrs[p] = sess_list[i].send(parties[p])

    with pytest.raises(ValueError):
        x * y


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
def test_bin_mul_private(get_clients, security):
    parties = get_clients(3)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)
    ring_size = 2
    bin_op = ReplicatedSharedTensor.get_op(ring_size, "mul")

    sh1 = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.bool)
    sh2 = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    shares1 = [sh1, sh1, sh1]
    shares2 = [sh2, sh2, sh2]
    rst_list1 = ReplicatedSharedTensor.distribute_shares(
        shares=shares1, session=session, ring_size=ring_size
    )
    rst_list2 = ReplicatedSharedTensor.distribute_shares(
        shares=shares2, session=session, ring_size=ring_size
    )
    tensor1 = MPCTensor(shares=rst_list1, session=session)
    tensor1.shape = sh1.shape
    tensor2 = MPCTensor(shares=rst_list2, session=session)
    tensor2.shape = sh2.shape

    secret1 = ReplicatedSharedTensor.shares_sum(shares1, ring_size)
    secret2 = ReplicatedSharedTensor.shares_sum(shares2, ring_size)

    result = operator.mul(tensor1, tensor2)
    expected_res = bin_op(secret1, secret2)

    assert (result.reconstruct(decode=False) == expected_res).all()


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
def test_prime_mul_private(get_clients, security):
    parties = get_clients(3)
    protocol = Falcon(security)
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)
    ring_size = PRIME_NUMBER
    prime_op = ReplicatedSharedTensor.get_op(ring_size, "mul")

    sh1 = torch.tensor([[32, 12, 23], [17, 35, 7]], dtype=torch.uint8)
    sh2 = torch.tensor([[45, 66, 47], [19, 57, 2]], dtype=torch.uint8)
    shares1 = [sh1, sh1, sh1]
    shares2 = [sh2, sh2, sh2]
    rst_list1 = ReplicatedSharedTensor.distribute_shares(
        shares=shares1, session=session, ring_size=ring_size
    )
    rst_list2 = ReplicatedSharedTensor.distribute_shares(
        shares=shares2, session=session, ring_size=ring_size
    )
    tensor1 = MPCTensor(shares=rst_list1, session=session)
    tensor1.shape = sh1.shape
    tensor2 = MPCTensor(shares=rst_list2, session=session)
    tensor2.shape = sh2.shape

    secret1 = ReplicatedSharedTensor.shares_sum(shares1, ring_size)
    secret2 = ReplicatedSharedTensor.shares_sum(shares2, ring_size)

    result = operator.mul(tensor1, tensor2)
    expected_res = prime_op(secret1, secret2)

    assert (result.reconstruct(decode=False) == expected_res).all()


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
def test_select_shares(get_clients, security) -> None:
    parties = get_clients(3)
    falcon = Falcon(security)
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    sh = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    shares = [sh, sh, sh]
    ptr_lst = ReplicatedSharedTensor.distribute_shares(shares, session, ring_size=2)
    b = MPCTensor(shares=ptr_lst, session=session, shape=sh.shape)

    x_val = torch.tensor([[1, 2], [3, 4]])
    y_val = torch.tensor([[5, 6], [7, 8]])
    x = MPCTensor(secret=x_val, session=session)
    y = MPCTensor(secret=y_val, session=session)

    z = Falcon.select_shares(x, y, b)

    expected_res = torch.tensor([[5.0, 2.0], [3.0, 8.0]])

    assert (expected_res == z.reconstruct()).all()


def test_select_shares_exception_ring(get_clients) -> None:
    parties = get_clients(3)
    falcon = Falcon()
    session = Session(parties=parties, protocol=falcon, ring_size=2 ** 32)
    SessionManager.setup_mpc(session)
    val = MPCTensor(secret=1, session=session)
    with pytest.raises(ValueError):
        Falcon.select_shares(val, val, val)


def test_select_shares_exception_shape(get_clients) -> None:
    parties = get_clients(3)
    falcon = Falcon()
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    val = MPCTensor(secret=1, session=session)
    rst = val.share_ptrs[0].get_copy()
    rst.ring_size = 2
    val.share_ptrs[0] = rst.send(parties[0])
    val.shape = None
    with pytest.raises(ValueError):
        Falcon.select_shares(val, val, val)


@pytest.mark.parametrize("security", ["semi-honest", "malicious"])
def test_private_compare(get_clients, security) -> None:
    parties = get_clients(3)
    falcon = Falcon(security_type=security)
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    base = session.config.encoder_base
    precision = session.config.encoder_precision
    fp_encoder = FixedPointEncoder(base=base, precision=precision)

    secret = torch.tensor([[358.85, 79.29], [67.78, 2415.50]])
    r = torch.tensor([[357.05, 90], [145.32, 2400.54]])
    r = fp_encoder.encode(r)
    x = MPCTensor(secret=secret, session=session)
    x_b = ABY3.bit_decomposition_ttp(x, session)  # bit shares
    x_p = []  # prime ring shares
    for share in x_b:
        x_p.append(ABY3.bit_injection(share, session, PRIME_NUMBER))

    tensor_type = get_type_from_ring(session.ring_size)
    result = Falcon.private_compare(x_p, r.type(tensor_type))
    expected_res = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    assert (result.reconstruct(decode=False) == expected_res).all()


def test_private_compare_exception(get_clients) -> None:
    parties = get_clients(3)
    falcon = Falcon()
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=1, session=session)
    r = torch.tensor([1])

    # Expection for not passing input tensor values as list.
    with pytest.raises(ValueError):
        Falcon.private_compare(x, r)

    # Exception for not passing a public value(torch.Tensor).
    with pytest.raises(ValueError):
        Falcon.private_compare([x], x)
