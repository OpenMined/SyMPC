# third party
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


def test_mul_private_integer(get_clients):

    # Not encoding because truncation hasn't been implemented yet for Falcon
    config = Config(encoder_base=1, encoder_precision=0)

    parties = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    result = tensor1 * tensor2

    assert (result.reconstruct() == (secret1 * secret2)).all()


def test_mul_private_integer_matrix(get_clients):

    # Not encoding because truncation hasn't been implemented yet for Falcon
    config = Config(encoder_base=1, encoder_precision=0)

    parties = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])

    secret2 = torch.tensor([[-1, 2, 3], [-9, 10, 1], [32, -23, 5]])

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    result = tensor1 * tensor2

    assert (result.reconstruct() == (secret1 * secret2)).all()


@pytest.mark.parametrize("parties", [2, 4])
def test_mul_private_exception_nothreeparties(get_clients, parties):

    # Not encoding because truncation hasn't been implemented yet
    config = Config(encoder_base=1, encoder_precision=0)

    parties = get_clients(parties)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    with pytest.raises(ValueError):
        tensor1 * tensor2


def test_mul_private_exception_malicious(get_clients):

    # Not encoding because truncation hasn't been implemented yet
    config = Config(encoder_base=1, encoder_precision=0)

    parties = get_clients(3)
    protocol = Falcon("malicious")
    session = Session(protocol=protocol, parties=parties, config=config)
    SessionManager.setup_mpc(session)

    secret1 = torch.tensor([[-100, 20, 30], [-90, 1000, 1], [1032, -323, 15]])
    secret2 = 8

    tensor1 = MPCTensor(secret=secret1, session=session)
    tensor2 = MPCTensor(secret=secret2, session=session)

    with pytest.raises(NotImplementedError):
        tensor1 * tensor2
