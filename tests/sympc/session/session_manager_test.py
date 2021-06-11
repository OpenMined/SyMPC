"""Tests for the SessionManager class."""

# stdlib
from uuid import UUID

# third party
import pytest

from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager


def test_session_manager_throw_exception_init():
    """Test correct initialisation of the SessionManager class."""
    # Test default init
    with pytest.raises(NotImplementedError):
        SessionManager()


def test_setup_mpc(get_clients):
    """Test setup_mpc method for session."""
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)
    assert isinstance(session.uuid, UUID)
    assert list(session.rank_to_uuid.keys()) == [0, 1]
    assert all(isinstance(e, UUID) for e in session.rank_to_uuid.values())


@pytest.mark.parametrize("parties", [2, 4, 5, 6])
def test_exception_parties_count(get_clients, parties) -> None:
    parties = get_clients(parties)
    protocol = Falcon("malicious")
    with pytest.raises(ValueError):
        session = Session(protocol=protocol, parties=parties)
        SessionManager.setup_mpc(session)
