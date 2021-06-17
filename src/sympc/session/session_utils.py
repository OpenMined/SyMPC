"""Function for accessing the current session."""
# stdlib
from typing import Optional
from typing import Union
from uuid import UUID

import sympc.session

from .session import Session


def get_session(uuid: Union[str, UUID]) -> Optional[Session]:
    """Gets the current session for a party as defined in the global space.

    Args:
        uuid (Union[UUID, str]): used to retrieve the session

    Returns:
        Session: MPC Session
    """
    return sympc.session.CURRENT_SESSION.get(str(uuid), None)


def set_session(session: Session) -> None:
    """Set the current sessionfor a party.

    Args:
        session (Session): session to be set

    """
    sympc.session.CURRENT_SESSION[str(session.uuid)] = session
