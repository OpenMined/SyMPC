from uuid import uuid1

from ..encoder import FixedPointEncoder

class Session:
    def __init__(self, parties, conf, ttp=None):
        self.fp_encoder = FixedPointEncoder(conf.fp_encoder_base, conf.fp_encoder_precision)
        self.uuid = uuid1

        # Each worker will have the rank as the index in the list
        self.parties = parties
        self.ttp = ttp

        self.crypto_store = {}

        self.protocol = None

        self.conf = conf
