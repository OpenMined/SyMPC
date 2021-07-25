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
from sympc.tensor import MPCTensor
from sympc.tensor import PRIME_NUMBER
from sympc.tensor import ReplicatedSharedTensor
from sympc.utils import get_nr_bits


def test_share_class() -> None:
    assert ABY3.share_class == ReplicatedSharedTensor


def test_session() -> None:
    protocol = ABY3("semi-honest")
    session = Session(protocol=protocol)
    assert type(session.protocol) == ABY3


def test_invalid_security_type():
    with pytest.raises(ValueError):
        ABY3(security_type="covert")


def test_eq():
    aby = ABY3()
    falcon1 = Falcon(security_type="malicious")
    falcon2 = Falcon()
    other2 = aby

    # Test equal protocol:
    assert aby == other2

    # Test different protocol security type
    assert aby != falcon1

    # Test different protocol objects
    assert aby != falcon2


def test_invalid_parties_trunc(get_clients) -> None:
    parties = get_clients(2)
    session = Session(parties=parties)

    with pytest.raises(ValueError):
        ABY3.truncate(None, session, 2 ** 32, None)


@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_truncation_algorithm1(get_clients, base, precision) -> None:
    parties = get_clients(3)
    falcon = Falcon("semi-honest")
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(parties=parties, protocol=falcon, config=config)
    SessionManager.setup_mpc(session)

    x = torch.tensor([[1.24, 4.51, 6.87], [7.87, 1301, 541]])

    x_mpc = MPCTensor(secret=x, session=session)

    result = ABY3.truncate(x_mpc, session, session.ring_size, session.config)

    fp_encoder = FixedPointEncoder(
        base=session.config.encoder_base, precision=session.config.encoder_precision
    )
    expected_res = x_mpc.reconstruct(decode=False) // fp_encoder.scale
    expected_res = fp_encoder.decode(expected_res)

    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


def test_invalid_mpc_pointer(get_clients) -> None:
    parties = get_clients(3)
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=1, session=session)
    # passing sharetensor pointer
    with pytest.raises(ValueError):
        ABY3.truncate(x, session, 2 ** 32, None)


@pytest.mark.parametrize("x1", ["zero", "one"])
@pytest.mark.parametrize("x2", ["zero", "one"])
@pytest.mark.parametrize("x3", ["zero", "one"])
@pytest.mark.parametrize("security_type", ["semi-honest", "malicious"])
def test_bit_injection_prime(get_clients, security_type, x1, x2, x3) -> None:
    parties = get_clients(3)
    falcon = Falcon(security_type=security_type)
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    ring_size = PRIME_NUMBER

    val = {
        "one": torch.tensor([1], dtype=torch.bool),
        "zero": torch.tensor([0], dtype=torch.bool),
    }

    shares = [val[x1], val[x2], val[x3]]  # All possible combinations
    ptr_lst = ReplicatedSharedTensor.distribute_shares(shares, session, ring_size=2)
    x = MPCTensor(shares=ptr_lst, session=session, shape=val["one"].shape)

    xbit = ABY3.bit_injection(x, session, ring_size)

    ring0 = int(xbit.share_ptrs[0].get_ring_size().get_copy())

    assert x.reconstruct(decode=False) == xbit.reconstruct(decode=False)
    assert ring_size == ring0


@pytest.mark.parametrize("x1", ["zero", "one"])
@pytest.mark.parametrize("x2", ["zero", "one"])
@pytest.mark.parametrize("x3", ["zero", "one"])
@pytest.mark.parametrize("security_type", ["semi-honest", "malicious"])
def test_bit_injection_session_ring(get_clients, security_type, x1, x2, x3) -> None:
    parties = get_clients(3)
    falcon = Falcon(security_type=security_type)
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    ring_size = session.ring_size

    val = {
        "one": torch.tensor([1], dtype=torch.bool),
        "zero": torch.tensor([0], dtype=torch.bool),
    }

    shares = [val[x1], val[x2], val[x3]]  # All possible combinations
    ptr_lst = ReplicatedSharedTensor.distribute_shares(shares, session, ring_size=2)
    x = MPCTensor(shares=ptr_lst, session=session, shape=val["one"].shape)

    xbit = ABY3.bit_injection(x, session, ring_size)

    ring0 = int(xbit.share_ptrs[0].get_ring_size().get_copy())

    assert x.reconstruct(decode=False) == xbit.reconstruct(decode=False)
    assert ring_size == ring0


@pytest.mark.parametrize("security_type", ["semi-honest", "malicious"])
def test_bit_decomposition_ttp(get_clients, security_type) -> None:
    parties = get_clients(3)
    falcon = Falcon(security_type=security_type)
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    secret = torch.tensor([[-1, 12], [-32, 45], [98, -5624]])
    x = MPCTensor(secret=secret, session=session)
    b_sh = ABY3.bit_decomposition_ttp(x, session)
    ring_size = x.session.ring_size
    tensor_type = x.session.tensor_type
    ring_bits = get_nr_bits(ring_size)

    val = 1
    expected_res = 0
    for i in range(ring_bits):
        if i != ring_bits - 1:
            expected_res += b_sh[i].reconstruct(decode=False).type(tensor_type) * val
        else:
            expected_res += b_sh[i].reconstruct(decode=False).type(tensor_type) * (-val)
        val *= 2
    assert (expected_res == x.reconstruct(decode=False)).all()
