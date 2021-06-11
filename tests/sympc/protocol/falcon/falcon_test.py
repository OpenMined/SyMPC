# third party
import pytest

from sympc.protocol import DefaultProtocol
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert Falcon.share_class == ReplicatedSharedTensor


def test_session() -> None:
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol)
    assert type(session.protocol) == Falcon


def test_invalid_security_type():
    with pytest.raises(ValueError):
        Falcon(security_type="covert")


def test_eq():
    falcon = Falcon()
    default1 = DefaultProtocol(security_type="malicious")
    default2 = DefaultProtocol()
    other2 = falcon

    # Test equal protocol:
    assert falcon == other2

    # Test different protocol security type
    assert falcon != default1

    # Test different protocol objects
    assert falcon != default2
