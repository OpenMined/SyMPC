from sympc.protocol import FALCON
from sympc.session import Session
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert FALCON.share_class == ReplicatedSharedTensor


def test_session() -> None:
    session = Session(protocol="FALCON")
    assert session.protocol == FALCON
