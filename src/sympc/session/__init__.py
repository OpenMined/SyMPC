"""Session class and utility functions used in conjunction with the session."""
# stdlib
from typing import Dict

from .session import Session
from .session_manager import SessionManager
from .session_utils import get_session
from .session_utils import set_session

# Mapping for rank -> Session
# rank is needed in the case we have VirtualMachines since we share the same workspace
CURRENT_SESSION: Dict[int, Session] = {}

__all__ = ["Session", "SessionManager", "get_session", "set_session"]
