# third party
import pytest
import torch

from sympc.protocol import FSS
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import ispointer


def test_share_tensor(get_clients) -> None:
    assert FSS.share_class == ShareTensor

    session = Session(parties=get_clients(3))
    SessionManager.setup_mpc(session)

    secret = torch.tensor([-1, 0, 1])
    shares = MPCTensor.generate_shares(secret, nr_parties=3)

    distributed_shares = FSS.distribute_shares(shares, session=session)

    assert all(ispointer(share_ptr) for share_ptr in distributed_shares)


def test_invalid_security_type():
    with pytest.raises(ValueError):
        FSS(security_type="malicious")


def test_eq():
    fss = FSS()
    falcon1 = Falcon(security_type="malicious")
    falcon2 = Falcon()
    other2 = fss

    # Test equal protocol:
    assert fss == other2

    # Test different protocol security type
    assert fss != falcon1

    # Test different protocol objects
    assert fss != falcon2
