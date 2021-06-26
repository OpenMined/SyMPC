# third party
import pytest

from sympc.protocol import ABY3
from sympc.session import Session
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
