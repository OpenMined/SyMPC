from sympc.protocol import FALCON
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert FALCON.share_class == ReplicatedSharedTensor
