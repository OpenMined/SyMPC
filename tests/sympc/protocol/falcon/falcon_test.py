# third party
import numpy as np

from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ReplicatedSharedTensor


def test_share_class() -> None:
    assert Falcon.share_class == ReplicatedSharedTensor


def test_session() -> None:
    session = Session(protocol="Falcon")
    assert session.protocol == Falcon


def test_session_distribute_reconstruct(get_clients) -> None:
    alice_client, bob_client, charles_client = get_clients(3)
    session = Session(
        protocol="Falcon", parties=[alice_client, bob_client, charles_client]
    )
    SessionManager.setup_mpc(session)

    secret = 42.32

    a = MPCTensor(secret=secret, session=session)

    assert np.allclose(secret, a.reconstruct(), atol=1e-5)
