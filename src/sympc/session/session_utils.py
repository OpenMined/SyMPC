"""Function for accessing the current session."""

import sympc.session

from .session import Session


def get_session() -> Session:
    """Gets the current session for a party as defined in the global space.

    Returns:
        Session: MPC Session
    """
    return sympc.session.current_session
