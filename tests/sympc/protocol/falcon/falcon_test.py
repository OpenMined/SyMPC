# third party
import pytest

from sympc.protocol import Falcon
from sympc.session import Session
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
