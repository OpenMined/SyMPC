from uuid import uuid1

from ..encoder import FixedPointEncoder

class Session:
    def __init__(self, parties, config, ttp=None):
        self.fp_encoder = FixedPointEncoder(config.fp_base)
        self.uuid = uuid1

        # Each worker will have the rank as the index in the list
        self.parties = parties
        self.ttp = ttp

        self.crypto_store = {}

        self.protocol = None
