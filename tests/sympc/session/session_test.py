"""
Tests for the Session class.
"""

# stdlib
from uuid import UUID
from uuid import uuid4

from sympc.config import Config
from sympc.session import Session
from sympc.session.utils import get_type_from_ring


def test_session_init():
    """
    Test correct initialisation of the Sessin class.
    """
    # Test default init
    session = Session()
    assert isinstance(session.uuid, UUID)
    assert session.parties == []
    assert session.trusted_third_party is None
    assert session.crypto_store == {}
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
    assert session.crypto_store == {}
    assert session.protocol is None
    assert session.config == config
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 32)
    assert session.ring_size == 2 ** 32
    assert session.min_value == -(2 ** 32) // 2
    assert session.max_value == (2 ** 32 - 1) // 2
