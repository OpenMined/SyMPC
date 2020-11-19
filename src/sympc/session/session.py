from uuid import uuid1

from ..config import Config
from ..encoder import FixedPointEncoder

class Session:
    def __init__(self, conf=None, parties=None, ttp=None, uuid=None):
        self.uuid = uuid1() if uuid is None else uuid

        # Each worker will have the rank as the index in the list
        self.parties = parties

        # Some protocols require a trusted third party
        # Ex: SPDZ
        self.trusted_third_party = ttp
        self.crypto_store = {}
        self.protocol = None
        self.conf = conf if conf else Config()

        # Those will be populated in the setup_mpc
        self.rank = None
        self.session_ptr = []

    def setup_mpc(self):
        raise NotImplementedError("Need to be able to send the session to other parties")
