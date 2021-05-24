from sympc.protocol import Falcon
from sympc.session import Session
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert Falcon.share_class == ReplicatedSharedTensor


def test_session() -> None:
    session = Session(protocol="Falcon")
    assert session.protocol == Falcon
