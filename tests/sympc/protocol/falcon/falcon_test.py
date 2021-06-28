# third party
import numpy as np
import pytest
import torch

from sympc.config import Config
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor


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


def test_mul_private(get_clients):
    parties = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties)
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


@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_mul_private_matrix(get_clients, base, precision):
    parties = get_clients(3)
    protocol = Falcon("semi-honest")
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


def test_mul_private_exception_malicious(get_clients):
    parties = get_clients(3)
    protocol = Falcon("malicious")
    session = Session(protocol=protocol, parties=parties)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    with pytest.raises(NotImplementedError):
        tensor1 * tensor2
