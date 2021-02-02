"""Tests for the Session class."""

# stdlib
from uuid import UUID
from uuid import uuid4

# third party
import torch

from sympc.config import Config
from sympc.session import Session
from sympc.session.utils import get_type_from_ring
from sympc.tensor import ShareTensor


def test_session_init():
    """Test correct initialisation of the Sessin class."""
    # Test default init
    session = Session()
    assert isinstance(session.uuid, UUID)
    assert session.parties == []
    assert session.trusted_third_party is None
    assert session.crypto_store is None
    assert session.protocol is None
    assert isinstance(session.config, Config)
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 64)
    assert session.ring_size == 2 ** 64
    assert session.min_value == -(2 ** 64) // 2
    assert session.max_value == (2 ** 64 - 1) // 2
    # Test custom init
    uuid = uuid4()
    config = Config()
    session = Session(
        parties=["alice", "bob"], ring_size=2 ** 32, config=config, ttp="TTP", uuid=uuid
    )
    assert session.uuid == uuid
    assert session.parties == ["alice", "bob"]
    assert session.trusted_third_party == "TTP"
    assert session.crypto_store is None
    assert session.protocol is None
    assert session.config == config
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 32)
    assert session.ring_size == 2 ** 32
    assert session.min_value == -(2 ** 32) // 2
    assert session.max_value == (2 ** 32 - 1) // 2


def test_przs_generate_random_share():
    """Test przs_generate_random_share method from Session."""
    session = Session()
    generators = [torch.Generator(), torch.Generator()]
    share = session.przs_generate_random_share(shape=(1, 2), generators=generators)
    assert isinstance(share, ShareTensor)
    assert (share.tensor == torch.tensor([0, 0])).all()


def test_eq():
    """Test __eq__ for Session."""
    session = Session()
    other1 = Session()
    other2 = session
    # Test different instances:
    assert session != 1
    # Test different sessions:
    assert session != other1
    # Test equal sessions:
    assert session == other2
