import torch


class AdditiveSharingTensor:

    def __init__(self, secret, session):
        self.session = session

        encoder = session.fp_encoder
        parties = session.parties

        secret = encoder.encode(secret)

        self.shares = torch.random

    def reconstruct(self):
        tensor = sum([
        for party in self.session.workers:
           pass
