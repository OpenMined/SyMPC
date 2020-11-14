import torch

from .utils import modular_to_real


class AdditiveSharingTensor:

    def __init__(self, secret=None, session=None):
        if not session:
            raise ValueError("Session should not be None")
        if secret is None:
            raise ValueError("Secret value should not be Non")

        self.session = session

        parties = session.parties
        encoder = session.fp_encoder
        secret = encoder.encode(secret)
        shares = AdditiveSharingTensor.generate_shares(secret, session)
        self.shares = []

        for share, party in zip(shares, parties):
            self.shares.append(share.send(party))

    @staticmethod
    def generate_shares(secret, session):
        parties = session.parties
        nr_parties = len(parties)

        shape = secret.shape
        min_value = session.conf.min_value
        max_value = session.conf.max_value

        random_shares = []
        for _ in range(nr_parties - 1):
            rand_long = torch.randint(min_value, max_value, shape).long()
            random_shares.append(rand_long)

        shares = []
        for i in range(len(parties)):
            if i == 0:
                share = random_shares[i]
            elif i < nr_parties - 1:
                share = random_shares[i] - random_shares[i-1]
            else:
                share = secret - random_shares[i-1]

            share = modular_to_real(share, session)
            shares.append(share)

        return shares

    def reconstruct(self):
        plaintext = self.shares[0].get()

        for share in self.shares[1:]:
            plaintext += share.get()

        plaintext = modular_to_real(plaintext, self.session)
        plaintext = self.session.fp_encoder.decode(plaintext)
        return plaintext


    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        for share in self.shares:
            out = f"{out}\n\t{share.client}->{share}"

        return out
