"""The implementation for the SessionManager.

This class holds the static methods used for the Session.
"""

# stdlib
import secrets
from typing import Dict
from uuid import UUID
from uuid import uuid4

from sympc.session.session import Session


class SessionManager:
    """Class used to manage sessions.

    Attributes:
        uuid (Optional[UUID]): used to identify a session
    """

    __slots__ = {
        "uuid",
    }

    def __init__(
        self,
    ) -> None:
        """Initializer for the Session Manager.

        Raises:
            NotImplementedError: This class it is not supposed to be instanciated.
        """
        raise NotImplementedError("This is not suposed to be instanciated!")

    @staticmethod
    def setup_mpc(session: Session) -> None:
        """Setup MPC.

        Must be called to send the session to all other parties involved in
        the computation.

        Args:
            session (Session): Session to send.
        """
        uuids: Dict[int, UUID] = {}
        for rank, party in enumerate(session.parties):
            # Assign a new rank before sending it to another party
            session_party = session.copy()
            session_party.rank = rank

            # And a new uuid
            session_party.uuid = uuid4()
            uuids[rank] = session_party.uuid
            session.session_ptrs.append(session_party.send(party))  # type: ignore

        session.uuid = uuid4()
        session.rank_to_uuid = uuids
        SessionManager._setup_przs(session)

    @staticmethod
    def _setup_przs(session: Session) -> None:
        """PRZS generator.

        Setup the Pseudo-Random-Zero-Share generators to the parties
        involved in the communication.

        Assume there are 3 parties:

        Step 1: Generate 3 seeds and send them in a ring like formation such that
        2 parties will generate the same random number at a given moment:
        - Party 1 holds G1 and G2
        - Party 2 holds G2 and G3
        - Party 3 holds G3 and G1

        Step 2: When they generate a PRZS:
            Party 1 generates: Next(G1) - Next(G2)
            Party 2 generates: Next(G2) - Next(G3)
            Party 3 generates: Next(G3) - Next(G1)
            -------------------------------------- +
                         PRZS: 0

        Step 3: The party that has the secret will add it to their own share

        Args:
            session (Session): Session involved in the communication.
        """
        seeds = [secrets.randbits(32) for _ in range(session.nr_parties)]

        for rank, remote_session in enumerate(session.session_ptrs):
            next_rank = (rank + 1) % session.nr_parties
            remote_session.init_generators(seeds[rank], seeds[next_rank])
