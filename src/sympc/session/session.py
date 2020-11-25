from uuid import uuid1
from typing import Union
from typing import List
from typing import Any

from sympc.config import Config
from sympc.session.utils import get_type_from_ring

import random


# TODO: This should not be here
import torch


class Session:
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
        "session_ptr",
        "ring_size",
        "tensor_type",
        "min_value",
        "max_value",
    }

    def __init__(
        self,
        parties: Union[List[Any], None] = None,
        ring_size: int = 2 ** 64,
        config: Union[None, Config] = None,
        ttp: Union[None] = None,
        uuid: Union[None, uuid1] = None,
    ) -> None:

        self.uuid = uuid1() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        # Only the party that is the CC (Control Center) will have access
        # to this
        self.parties = parties

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp

        self.crypto_store = {}
        self.protocol = None
        self.config = config if config else Config()

        self.przs_generators = None

        # Those will be populated in the setup_mpc
        self.rank = None
        self.session_ptr = []

        # Ring size
        self.tensor_type = get_type_from_ring(ring_size)
        self.ring_size = ring_size
        self.min_value = -(ring_size) // 2
        self.max_value = (ring_size - 1) // 2

    def przs_generate_random_share(
        self, shape: Union[tuple, torch.Size], generators: List[torch.Generator]
    ) -> "ShareTensor":

        from sympc.tensor import ShareTensor

        gen0, gen1 = generators

        current_share = torch.randint(
            low=self.min_value,
            high=self.max_value,
            size=shape,
            dtype=self.tensor_type,
            generator=gen0,
        )

        next_share = torch.randint(
            low=self.min_value,
            high=self.max_value,
            size=shape,
            dtype=self.tensor_type,
            generator=gen1,
        )

        share = ShareTensor(session=self)
        share.tensor = current_share - next_share

        return share

    @staticmethod
    def setup_mpc(session: "Session") -> None:
        for rank, party in enumerate(session.parties):
            # Assign a new rank before sending it to another party
            session.rank = rank
            session.session_ptr.append(session.send(party))

        Session._setup_przs(session)

    @staticmethod
    def _setup_przs(session: "Session") -> None:
        nr_parties = len(session.parties)

        # Create the remote lists where we add the generators
        session.przs_generators = [
            party.python.List([None, None]) for party in session.parties
        ]

        for rank in range(nr_parties):

            # TODO: Need to change this with something secure - see CSPRNG
            seed = random.randint(-(2 ** 31), 2 ** 31)
            next_rank = (rank + 1) % nr_parties

            gen_current = session.parties[rank].torch.Generator()
            gen_current.manual_seed(seed)

            gen_next = session.parties[next_rank].torch.Generator()
            gen_next.manual_seed(seed)

            session.przs_generators[rank][1] = gen_current
            session.przs_generators[next_rank][0] = gen_next
