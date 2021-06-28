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
from sympc.tensor import ReplicatedSharedTensor


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
        ABY3.truncate(None, session)


@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_truncation_algorithm1(get_clients, base, precision) -> None:
    parties = get_clients(3)
    falcon = Falcon("semi-honest")
    config = Config(encoder_base=base, encoder_precision=precision)
    session = Session(parties=parties, protocol=falcon, config=config)
    SessionManager.setup_mpc(session)

    x = torch.tensor([[1.24, 4.51, 6.87], [7.87, 1301, 541]])

    x_mpc = MPCTensor(secret=x, session=session)

    result = ABY3.truncate(x_mpc, session)

    fp_encoder = FixedPointEncoder(
        base=session.config.encoder_base, precision=session.config.encoder_precision
    )
    expected_res = x_mpc.reconstruct(decode=False) // fp_encoder.scale
    expected_res = fp_encoder.decode(expected_res)

    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)


def test_invalid_parties(get_clients) -> None:
    parties = get_clients(2)
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=1, session=session)
    with pytest.raises(ValueError):
        ABY3.truncate(x, session)


def test_invalid_mpc_pointer(get_clients) -> None:
    parties = get_clients(3)
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)
    x = MPCTensor(secret=1, session=session)
    # passing sharetensor pointer
    with pytest.raises(ValueError):
        ABY3.truncate(x, session)
