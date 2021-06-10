from sympc.protocol import DefaultProtocol
from sympc.session import Session
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert DefaultProtocol.share_class == ReplicatedSharedTensor


def test_session() -> None:
    defaultprotocol = DefaultProtocol()
    session = Session(protocol=defaultprotocol)
    assert type(session.protocol) == DefaultProtocol
