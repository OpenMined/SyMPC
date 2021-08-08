"""The implementation for the Session.

It is used to identify a MPC computation done between multiple parties.

Example:
    Alice Bob and John wants to do some computation
    Alice John and Beatrice also wants to do some computation

    The resources/config Alice uses for the first computation should be
    isolated and should not disturb the second computation
"""


# stdlib
from copy import deepcopy
import operator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID

# third party
# TODO: This should not be here
import torch

from sympc.config import Config
from sympc.protocol.protocol import Protocol
from sympc.utils import generate_random_element
from sympc.utils import get_new_generator
from sympc.utils import get_type_from_ring


class Session:
    """Class used to keep information about computation done in SMPC.

    Attributes:
        id (UID): The id to store the session
        tags (Optional[List[str]): an optional list of strings that are tags used at search
        description (Optional[str]): an optional string used to describe the session

        uuid (Optional[UUID]): used to identify a session
        rank_to_uuid (Dict[int, UUID]): used by the orchestrator to keep track of the session
            uuid for each party
        parties (Optional[List[Any]): used to send/receive messages
        nr_parties (int): number of parties
        trusted_third_party (Optional[Any]): the trusted third party
        crypto_store (CryptoStore): keep track of items needed in MPC (for the moment not used)
        protocol (Optional[str]): specify what protocol to register for a session
        config (Config): used for the Fixed Precision Encoder
        przs_generator (Optional[torch.Generator]): Pseudo-Random-Zero-Share Generators
            pointers to the parties generators
        rank (int): Rank for a party in this session
        session_ptrs (List[Session]): pointers to the session that should be identical to the
            one we have
        ring_size (int): field used for the operations applied on the shares
        min_value (int): the minimum value allowed for tensors' values
        max_value (int): the maximum value allowed for tensors' values
        tensor_type (Union[torch.dtype): tensor type used in the computation, this is used
            such that we get the "modulo" operation for free
    """

    # Those values are not used at comparison
    NOT_COMPARE = {
        "id",
        "description",
        "tags",
        "parties",
        "crypto_store",
        "przs_generators",
        "session_ptrs",
    }

    __slots__ = {
        # Populated in Syft
        "id",
        "tags",
        "description",
        "uuid",
        "rank_to_uuid",
        "parties",
        "nr_parties",
        "trusted_third_party",
        "crypto_store",
        "protocol",
        "config",
        "przs_generators",
        "session_ptrs",
        "rank",
        "ring_size",
        "min_value",
        "max_value",
        "tensor_type",
        "autograd_active",
    }

    def __init__(
        self,
        parties: Optional[List[Any]] = None,
        ring_size: int = 2 ** 64,
        config: Optional[Config] = None,
        protocol: Optional[Protocol] = None,
        ttp: Optional[Any] = None,
    ) -> None:
        """Initializer for the Session.

        Args:
            parties (Optional[List[Any]): Used to send/receive messages:
            ring_size (int): Field used for the operations applied on the shares
            config (Optional[Config]): Configuration used for information needed
                by the Fixed Point Encoder. Defaults None
            protocol (Optional[str]): Protocol. Defaults None
            ttp (Optional[Any]): Trusted third party. Defaults None.

        Raises:
            ValueError: If protocol is not registered.
        """
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
        self.crypto_store: Optional[
            Dict[Any, Any]
        ] = None  # TODO: this should be CryptoStore

        self.protocol: Protocol = None
        if protocol is None:
            self.protocol = Protocol.registered_protocols["FSS"]()
        else:
            if type(protocol).__name__ not in Protocol.registered_protocols:
                raise ValueError(f"{type(protocol).__name__} not registered!")

            self.protocol = protocol

        if (
            self.parties
            and len(self.parties) < 3
            and self.protocol.security_type == "malicious"
        ):

            raise ValueError(
                "Malicious security cannot be provided to less than 3 parties"
            )

        self.config = config if config else Config()

        self.przs_generators: List[Optional[torch.Generator]] = []

        # Those will be populated in the setup_mpc
        self.rank: int = -1
        self.uuid: Optional[UUID] = None
        self.session_ptrs = []

        self.rank_to_uuid: Dict[int, UUID] = {}

        # Ring size
        self.tensor_type: Union[torch.dtype] = get_type_from_ring(ring_size)
        self.ring_size = ring_size
        self.min_value = -(ring_size) // 2
        self.max_value = (ring_size - 1) // 2

        self.autograd_active = False

    def get_protocol(self) -> Protocol:
        """Get protocol.

        Returns:
            Protocol
        """
        return self.protocol

    def _generate_random_share(
        self, shape: Union[tuple, torch.Size], ring_size: int
    ) -> List[torch.Tensor]:
        """Generate random tensor share for the given shape and ring_size.

        Args:
            shape (Union[tuple, torch.Size]): Shape for the share.
            ring_size (int): ring size to generate share.

        The generators are invoked in Counter(CTR) mode as parties with the same initial seeds
        could generate correlated random numbers on subsequent invocations.

        Returns:
            List[torch.Tensor] : shares generated by the generators.
        """
        from sympc.tensor import PRIME_NUMBER

        gen1, gen2 = self.przs_generators
        tensor_type = get_type_from_ring(ring_size)
        max_val = PRIME_NUMBER if ring_size == PRIME_NUMBER else None

        tensor_share1 = generate_random_element(
            tensor_type=tensor_type, generator=gen1, shape=shape, max_val=max_val
        )

        tensor_share2 = generate_random_element(
            tensor_type=tensor_type, generator=gen2, shape=shape, max_val=max_val
        )

        return [tensor_share1, tensor_share2]

    def przs_generate_random_share(
        self,
        shape: Union[tuple, torch.Size],
        ring_size: Optional[str] = None,
    ) -> Any:
        """Generates a random zero share using the two generators held by a party.

        Args:
            shape (Union[tuple, torch.Size]): Shape for the share.
            ring_size (str): ring size to generate share.

        Returns:
            Any: ShareTensor or ReplicatedSharedTensor

        """
        from sympc.tensor import ReplicatedSharedTensor
        from sympc.tensor import ShareTensor

        if ring_size is None:
            ring_size = self.ring_size
        else:
            ring_size = int(ring_size)  # 2**64 cannot be serialized.

        current_share, next_share = self._generate_random_share(shape, ring_size)

        if self.protocol.share_class == ShareTensor:
            # It has encoder_precision = 0 such that the value would not be encoded
            share = ShareTensor(
                data=current_share - next_share,
                session_uuid=self.uuid,
                config=Config(encoder_precision=0),
                ring_size=ring_size,
            )
        else:
            op = ReplicatedSharedTensor.get_op(ring_size, "sub")
            share = ReplicatedSharedTensor(
                shares=[op(current_share, next_share)],
                session_uuid=self.uuid,
                config=Config(encoder_precision=0),
                ring_size=ring_size,
            )
        return share

    def prrs_generate_random_share(
        self,
        shape: Union[tuple, torch.Size],
        ring_size: Optional[str] = None,
    ) -> Any:
        """Generates a random share using the generators held by a party.

        Args:
            shape (Union[tuple, torch.Size]): Shape for the share.
            ring_size (str): ring size to generate share.

        Returns:
            Any: ShareTensor or ReplicatedSharedTensor

        """
        from sympc.tensor import ReplicatedSharedTensor
        from sympc.tensor import ShareTensor

        if ring_size is None:
            ring_size = self.ring_size
        else:
            ring_size = int(ring_size)  # 2**64 cannot be serialized.

        share1, share2 = self._generate_random_share(shape, ring_size)

        if self.protocol.share_class == ShareTensor:
            # It has encoder_precision = 0 such that the value would not be encoded
            share = ShareTensor(
                data=share1,
                session_uuid=self.uuid,
                config=Config(encoder_precision=0),
                ring_size=ring_size,
            )
        else:
            share = ReplicatedSharedTensor(
                shares=[share1, share2],
                session_uuid=self.uuid,
                config=Config(encoder_precision=0),
                ring_size=ring_size,
            )

        return share

    def init_generators(self, seed_current: int, seed_next: int) -> None:
        """Initialize the generators - that are used for Pseudo Random Zero Shares.

        Args:
            seed_current (int): the seed for our party
            seed_next (int): thee seed for the next party
        """
        generator_current = get_new_generator(seed_current)
        generator_next = get_new_generator(seed_next)
        self.przs_generators = [generator_current, generator_next]

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of attributes to compare.

        Args:
            other (Any): Session to compare.

        Returns:
            Bool. True if equal False if not.
        """
        if not isinstance(other, self.__class__):
            return False

        attr_getters = [
            operator.attrgetter(attr) for attr in self.__slots__ - Session.NOT_COMPARE
        ]
        return all(getter(self) == getter(other) for getter in attr_getters)

    def copy(self) -> "Session":
        """Copy specific fields from the session.

        Returns:
            A copy of the current Session.
        """
        session = Session()
        session.nr_parties = self.nr_parties
        session.config = deepcopy(self.config)
        session.protocol = self.protocol
        session.ring_size = self.ring_size

        return session
