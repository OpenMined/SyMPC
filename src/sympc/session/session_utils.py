"""Function for accessing the current session."""
# stdlib
from typing import Optional

import sympc.session

from .session import Session


def get_session(uuid_str: str) -> Optional[Session]:
    """Gets the current session for a party as defined in the global space.

    Args:
        uuid_str (str): used to retrieve the session

    Returns:
        Session: MPC Session
    """
    return sympc.session.CURRENT_SESSION.get(uuid_str, None)


def set_session(session: Session) -> None:
    """Set the current sessionfor a party.

    Args:
        session (Session): session to be set

    """
    sympc.session.CURRENT_SESSION[str(session.uuid)] = session
