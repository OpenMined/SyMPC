# third party
import pytest

from sympc.protocol import DefaultProtocol
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert DefaultProtocol.share_class == ReplicatedSharedTensor


def test_session() -> None:
    defaultprotocol = DefaultProtocol()
    session = Session(protocol=defaultprotocol)
    assert type(session.protocol) == DefaultProtocol


def test_invalid_security_type():
    with pytest.raises(ValueError):
        DefaultProtocol(security_type="covert")


def test_eq():
    default = DefaultProtocol()
    falcon1 = Falcon(security_type="malicious")
    falcon2 = Falcon()
    other2 = default

    # Test equal protocol:
    assert default == other2

    # Test different protocol security type
    assert default != falcon1

    # Test different protocol objects
    assert default != falcon2
