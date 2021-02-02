"""Tests for the SessionManager class."""

# stdlib
from uuid import UUID
from uuid import uuid4

from sympc.session import Session
from sympc.session import SessionManager


def test_session_manager_init():
    """Test correct initialisation of the SessionManager class."""
    # Test default init
    session = SessionManager()
    assert isinstance(session.uuid, UUID)
    # Test custom init
    uuid = uuid4()
    session = Session(uuid=uuid)
    assert session.uuid == uuid


def test_setup_mpc(get_clients):
    """Test setup_mpc method for session."""
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)
    assert session.rank == 1
    assert len(session.session_ptrs) == 2


def test_setup_przs(get_clients):
    """Test _setup_przs method for session."""
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager._setup_przs(session)
    assert True


def test_eq():
    """Test __eq__ for SessionManager."""
    session_manager = SessionManager()
    other1 = SessionManager()
    other2 = session_manager
    # Test different instances:
    assert session_manager != 1
    # Test different session_managers:
    assert session_manager != other1
    # Test equal session_managers:
    assert session_manager == other2
