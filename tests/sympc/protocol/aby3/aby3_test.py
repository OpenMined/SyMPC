# third party
import numpy as np
import pytest
import torch

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


def test_invalid_parties_trunc(get_clients) -> None:
    parties = get_clients(2)
    session = Session(parties=parties)

    with pytest.raises(ValueError):
        ABY3.truncate(None, session)


def test_trunc1(get_clients) -> None:
    parties = get_clients(3)
    falcon = Falcon()
    session = Session(parties=parties, protocol=falcon)
    SessionManager.setup_mpc(session)

    x = torch.tensor([[1.24, 4.51, 6.87], [7.87, 1301, 541]])

    x_mpc = MPCTensor(secret=x, session=session)

    x_trunc = ABY3.trunc1(x_mpc.share_ptrs, x_mpc.shape, session)
    result = MPCTensor(shares=x_trunc, session=session)

    fp_encoder = FixedPointEncoder(
        base=session.config.encoder_base, precision=session.config.encoder_precision
    )
    expected_res = x_mpc.reconstruct(decode=False) // fp_encoder.scale
    expected_res = fp_encoder.decode(expected_res)

    assert np.allclose(result.reconstruct(), expected_res, atol=1e-3)
