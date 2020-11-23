from uuid import uuid1

from ..config import Config
from copy import deepcopy

class Session:
    def __init__(self, config=None, parties=None, ttp=None, uuid=None):
        self.uuid = uuid1() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        self.parties = parties

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp
        self.crypto_store = {}
        self.protocol = None
        self.config = config if config else Config()

        # Those will be populated in the setup_mpc
        self.rank = None
        self.session_ptr = []

    def get_copy(self):
        session_copy = Session()
        session_copy.uuid = deepcopy(self.uuid)
        session_copy.parties = [party for party in self.parties]
        session_copy.trusted_third_party = self.trusted_third_party
        session_copy.crypto_store = {}
        session_copy.protocol = self.protocol
        session_copy.config = deepcopy(self.config)
        session_copy.rank = self.rank
        session_copy.session_ptr = [s_ptr for s_ptr in self.session_ptr]

        return session_copy

    def setup_mpc(self):
        raise NotImplementedError("Need to be able to send the session to other parties")
