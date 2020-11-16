import torch

from .utils import modular_to_real
from ..encoder import FixedPointEncoder
import operator


class AdditiveSharingTensor:

    def __init__(self, secret=None, shares=None, encoder_base=10, encoder_precision=4, session=None):
        if not session:
            raise ValueError("Session should not be None")

        self.session = session
        self.shape = None

        self.fp_encoder = FixedPointEncoder(base=encoder_base, precision=encoder_precision)

        if secret is not None:
            parties = session.parties
            self.shape = secret.shape
            secret = self.fp_encoder.encode(secret)
            shares = AdditiveSharingTensor.generate_shares(secret, session)
            self.shares = []
            for share, party in zip(shares, parties):
                self.shares.append(share.send(party))
        elif shares is not None:
            self.shares = shares
        else:
            raise ValueError("Shares or Secret should not be None")

    @staticmethod
    def from_shares(shares, session):
        return AdditiveSharingTensor(shares=shares, session=session)

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
        plaintext = self.fp_encoder.decode(plaintext)
        return plaintext


    def add(self, y):
        return self.__apply_op(y, "add")

    def sub(self, y):
        return self.__apply_op(y, "sub")

    def mul(self, y):
        return self.__apply_op(y, "mul")

    def div(self, y):
        return self.__apply_op(y, "div")

    def __apply_private_op(self, y, op_str):
        from ..protocol import SPDZ
        if y.session.uuid != self.session.uuid:
            raise ValueError(f"Need same session {self.session.uuid} and {y.session.uuid}")
        if op_str == "mul":
            result = SPDZ.mul_master(self, y)
            result.shares = [modular_to_real(share, self.session) for share in result.shares]
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            shares = [modular_to_real(op(*share_tuple), self.session) for share_tuple in zip(self.shares, y.shares)]
            result = AdditiveSharingTensor.from_shares(shares=shares, session=self.session)

        return result

    def __apply_public_op(self, y, op_str):
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor([y])

        y = self.fp_encoder.encode(y)

        op = getattr(operator, op_str)
        # Here are two sens: one for modular_to_real and one for op
        # TODO: Make only one operation

        if op_str == "mul":
            shares = [modular_to_real(op(share, y) // self.fp_encoder.scale, self.session) for share in self.shares]
        else:
            operands_shares = [y, 0, 0]
            shares = [modular_to_real(op(*tuple_shares), self.session) for tuple_shares in zip(self.shares, operands_shares)]


        result = AdditiveSharingTensor.from_shares(shares=shares, session=self.session)
        return result

    def __apply_op(self, y, op):
        is_private = isinstance(y, AdditiveSharingTensor)

        if is_private:
            return self.__apply_private_op(y, op)

        return self.__apply_public_op(y, op)

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        for share in self.shares:
            out = f"{out}\n\t{share.client} -> {share}"

        return out

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
