"""Tests for the Session class."""

# stdlib

# third party
import pytest
import torch

from sympc.config import Config
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import ShareTensor
from sympc.utils import get_new_generator
from sympc.utils import get_type_from_ring


def test_session_default_init() -> None:
    """Test correct initialisation of the Sessin class."""
    # Test default init
    session = Session()
    assert session.uuid is None
    assert session.parties == []
    assert session.trusted_third_party is None
    assert session.crypto_store is None
    assert session.protocol is not None
    assert isinstance(session.config, Config)
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 64)
    assert session.ring_size == 2 ** 64
    assert session.min_value == -(2 ** 64) // 2
    assert session.max_value == (2 ** 64 - 1) // 2


def test_session_custom_init() -> None:
    config = Config()
    session = Session(
        parties=["alice", "bob"], ring_size=2 ** 32, config=config, ttp="TTP"
    )
    assert session.uuid is None
    assert session.parties == ["alice", "bob"]
    assert session.trusted_third_party == "TTP"
    assert session.crypto_store is None
    assert session.protocol is not None
    assert session.config == config
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 32)
    assert session.ring_size == 2 ** 32
    assert session.min_value == -(2 ** 32) // 2
    assert session.max_value == (2 ** 32 - 1) // 2


def test_przs_generate_random_share(get_clients) -> None:
    """Test przs_generate_random_share method from Session."""
    session = Session()
    SessionManager.setup_mpc(session)
    gen1 = get_new_generator(42)
    gen2 = get_new_generator(43)
    session.przs_generators = [gen1, gen2]
    share = session.przs_generate_random_share(shape=(2, 1))
    assert isinstance(share, ShareTensor)
    target_tensor = torch.tensor(([-1540733531777602634], [2813554787685566880]))
    assert (share.tensor == target_tensor).all()


def test_eq() -> None:
    """Test __eq__ for Session."""
    session = Session()
    other1 = Session()
    other2 = session

    # Test different instances:
    assert session != 1

    # Test equal sessions:
    assert session == other2

    # Test same sessions (until we call setup mpc):
    assert session == other1

    SessionManager.setup_mpc(session)

    assert session != other1


def test_copy() -> None:
    session = Session()

    copy_session = session.copy()

    assert session.nr_parties == copy_session.nr_parties
    assert session.config == copy_session.config
    assert session.protocol == copy_session.protocol


def test_invalid_protocol_exception() -> None:
    with pytest.raises(ValueError):
        Session(protocol="fs")


def test_invalid_ringsize_exception() -> None:
    with pytest.raises(ValueError):
        Session(ring_size=2 ** 63)
