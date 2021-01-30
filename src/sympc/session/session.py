"""The implementation for the Session It is used to identify a MPC computation
done between multiple parties.

This would be used in case a party is involved in multiple MPC session,
this one is used to identify in which one is used

Example:
    Alice Bob and John wants to do some computation
    Alice John and Beatrice also wants to do some computation

    The resources/config Alice uses for the first computation should be
    isolated and should not disturb the second computation
"""


# stdlib
import operator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID
from uuid import uuid4

# third party
# TODO: This should not be here
import torch

from sympc.config import Config
from sympc.session.utils import generate_random_element
from sympc.session.utils import get_type_from_ring


class Session:
    """Class used to keep information about computation done in SMPC.

    Arguments:
        parties (Optional[List[Any]): used to send/receive messages
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
        nr_parties (int): number of parties
        trusted_third_party (Optional[Any]): the trusted third party
        crypto_store (CryptoStore): keep track of items needed in MPC (for the moment not used)
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
    NOT_COMPARE = {"id", "description", "tags", "parties", "crypto_store"}

    __slots__ = {
        # Populated in Syft
        "id",
        "tags",
        "description",
        "uuid",
        "parties",
        "nr_parties",
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
        parties: Optional[List[Any]] = None,
        ring_size: int = 2 ** 64,
        config: Optional[Config] = None,
        ttp: Optional[Any] = None,
        uuid: Optional[UUID] = None,
    ) -> None:
        """Initializer for the Session."""

        self.uuid = uuid4() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        # Only the party that is the CC (Control Center) will have access
        # to this

        self.parties: List[Any]
        self.nr_parties: int

        if parties is None:
            self.parties = []
            self.nr_parties = 0
        else:
            self.parties = parties
            self.nr_parties = len(parties)

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp

        # The CryptoStore is initialized at each party when it is unserialized
        self.crypto_store: Dict[Any, Any] = None  # TODO: this should be CryptoStore

        self.protocol: Optional[str] = None

        self.config = config if config else Config()

        self.przs_generators: List[List[torch.Generator]] = []

        # Those will be populated in the setup_mpc
        self.rank = -1
        self.session_ptrs: List[Session] = []

        # Ring size
        self.tensor_type: Union[torch.dtype] = get_type_from_ring(ring_size)
        self.ring_size = ring_size
        self.min_value = -(ring_size) // 2
        self.max_value = (ring_size - 1) // 2

    def populate_crypto_store(
        self, op_str: str, primitives: Any, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> None:
        self.crypto_store.populate_store(op_str, primitives, *args, **kwargs)

    def przs_generate_random_share(
        self, shape: Union[tuple, torch.Size], generators: List[torch.Generator]
    ) -> Any:
        """Generate a random share using the two generators that are hold by a
        party."""

        from sympc.tensor import ShareTensor

        gen0, gen1 = generators

        current_share = generate_random_element(
            tensor_type=self.tensor_type,
            generator=gen0,
            shape=shape,
        )

        next_share = generate_random_element(
            tensor_type=self.tensor_type,
            generator=gen1,
            shape=shape,
        )

        share = ShareTensor(session=self)
        share.tensor = current_share - next_share

        return share

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of
        attributes to compare.

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
