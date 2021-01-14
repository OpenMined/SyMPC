"""
The implementation for the SessionManager.

This class holds the static methods used for the Session.
"""

# stdlib
import operator
import secrets
from typing import Any
from typing import Dict
from typing import Optional
from uuid import UUID
from uuid import uuid4

from sympc.session.session import Session


class SessionManager:
    """
    Class used to manage sessions.

    Arguments:
        ring_size (int): field used for the operations applied on the shares
        config (Optional[Config]): configuration used for information needed
            by the Fixed Point Encoder
        ttp (Optional[Any]): trusted third party
        uuid (Optional[UUID]): used to identify a session

    Attributes:
        Syft Serializable Attributes

        id (UID): the id to store the session
        tags (Optional[List[str]): an optional list of strings that are tags used at search
        description (Optional[str]): an optional string used to describe the session


        uuid (Optional[UUID]): used to identify a session
        parties (Optional[List[Any]): used to send/receive messages
        trusted_third_party (Optional[Any]): the trusted third party
        crypto_store (Dict[Any, Any]): keep track of items needed in MPC (for the moment not used)
        protocol (Optional[str]): specify what protocol to register for a session
        config (Config): used for the Fixed Precision Encoder
        przs_generator (Optional[torch.Generator]): Pseudo-Random-Zero-Share Generators
            pointers to the parties generators
        rank (int): Rank for a party in this session
        sessio_ptrs (List[Session]): pointers to the session that should be identical to the
            one we have
        ring_size (int): field used for the operations applied on the shares
        min_value (int): the minimum value allowed for tensors' values
        max_value (int): the maximum value allowed for tensors' values
        tensor_type (Union[torch.dtype): tensor type used in the computation, this is used
            such that we get the "modulo" operation for free
    """

    # Those values are not used at comparison
    NOT_COMPARE = {"id", "description", "tags", "parties"}

    __slots__ = {
        # Populated in Syft
        "id",
        "tags",
        "description",
        "uuid",
        "parties",
        "trusted_third_party",
        "crypto_store",
        "protocol",
        "config",
        "przs_generators",
        "rank",
        "session_ptrs",
        "ring_size",
        "min_value",
        "max_value",
        "tensor_type",
    }

    def __init__(
        self,
        ttp: Optional[Any] = None,
        uuid: Optional[UUID] = None,
    ) -> None:
        """ Initializer for the Session """

        self.uuid = uuid4() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        # Only the party that is the CC (Control Center) will have access
        # to this

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp

        self.crypto_store: Dict[Any, Any] = {}
        self.protocol: Optional[str] = None

    @staticmethod
    def setup_mpc(session: "Session") -> None:
        """Must be called to send the session to all other parties involved in the
        computation.
        """
        for rank, party in enumerate(session.parties):
            # Assign a new rank before sending it to another party
            session.rank = rank
            session.session_ptrs.append(session.send(party))  # type: ignore

        SessionManager._setup_przs(session)

    @staticmethod
    def _setup_przs(session: "Session") -> None:
        """Setup the Pseudo-Random-Zero-Share generators to the parties involved
        in the communication.

        Assume there are 3 parties:

        Step 1: Generate 3 seeds and send them in a ring like formation such that
        2 parties will generate the same random number at a given moment:
        - Party 1 holds G1 and G2
        - Party 2 holds G2 and G3
        - Party 3 holds G3 and

        Step 2: When they generate a PRZS:
            Party 1 generates: Next(G1) - Next(G2)
            Party 2 generates: Next(G2) - Next(G3)
            Party 3 generates: Next(G3) - Next(G1)
            -------------------------------------- +
                         PRZS: 0

        Step 3: The party that has the secret will add it to their own share
        """
        nr_parties = len(session.parties)

        # Create the remote lists where we add the generators
        session.przs_generators = [
            party.python.List([None, None]) for party in session.parties
        ]

        parties = session.parties

        for rank in range(nr_parties):
            seed = secrets.randbits(32)
            next_rank = (rank + 1) % nr_parties

            gen_current = session.parties[rank].sympc.session.get_generator(seed)
            gen_next = parties[next_rank].sympc.session.get_generator(seed)

            session.przs_generators[rank][1] = gen_current
            session.przs_generators[next_rank][0] = gen_next

    def __eq__(self, other: Any) -> bool:
        """
        Check if "self" is equal with another object given a set of attributes
        to compare.

        :return: if self and other are equal
        :rtype: bool
        """
        if not isinstance(other, self.__class__):
            return False

        if self.__slots__ != other.__slots__:
            return False

        attr_getters = [
            operator.attrgetter(attr) for attr in self.__slots__ - Session.NOT_COMPARE
        ]
        return all(getter(self) == getter(other) for getter in attr_getters)
