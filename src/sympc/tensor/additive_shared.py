import torch

from .utils import modulo
from ..encoder import FixedPointEncoder
from copy import deepcopy
import operator


class AdditiveSharingTensor:

    def __init__(self, secret=None, shares=None, session=None):
        if not session:
            raise ValueError("Session should not be None")

        if len(session.session_ptr) == 0:
            raise ValueError("setup_mpc was not called on the session")

        self.session = session
        self.shape = None

        conf = session.config
        self.fp_encoder = FixedPointEncoder(base=conf.enc_base, precision=conf.enc_precision)

        if secret is not None:
            parties = session.parties
            secret = self.fp_encoder.encode(secret)
            self.shape = secret.shape
            shares = AdditiveSharingTensor.generate_shares(secret, session)
            self.shares = []
            for share, party in zip(shares, parties):
                self.shares.append(share.send(party))
        elif shares is not None:
            self.shares = shares

    @staticmethod
    def generate_shares(secret, session):
        parties = session.parties
        nr_parties = len(parties)

        shape = secret.shape
        min_value = session.config.min_value
        max_value = session.config.max_value

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

            share = modulo(share, session)
            shares.append(share)

        return shares

    def reconstruct(self, decode=True):
        plaintext = self.shares[0].get()

        for share in self.shares[1:]:
            plaintext = modulo(plaintext + share.get(), self.session)

        if decode:
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
        if y.session.uuid != self.session.uuid:
            raise ValueError(f"Need same session {self.session.uuid} and {y.session.uuid}")

        if op_str in {"mul"}:
            from ..protocol import spdz

            shares = spdz.mul_master(self, y, op_str)
            self_precision = self.fp_encoder.precision
            y_precision = y.fp_encoder.precision
            result = AdditiveSharingTensor.__apply_encoding(self_precision, y_precision, shares, self.session, op_str)
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            result = AdditiveSharingTensor(session=self.session)
            result.encoder = deepcopy(self.fp_encoder)
            result.shares = [
                    modulo(op(*share_tuple), self.session)
                    for share_tuple in zip(self.shares, y.shares)
            ]

        return result

    @staticmethod
    def __apply_encoding(x_precision, y_precision, shares, session, op_str):
        max_prec = max(x_precision, y_precision)
        fp_encoder = FixedPointEncoder(precision=max_prec)

        if x_precision and y_precision:
                shares = [share // fp_encoder.scale for share in shares]

        shares = [modulo(share, session) for share in shares]

        result = AdditiveSharingTensor(shares=shares, session=session)
        result.fp_encoder = fp_encoder

        return result

    def __apply_public_op(self, y, op_str):
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor([y])

        y = self.fp_encoder.encode(y)

        op = getattr(operator, op_str)
        # Here are two sens: one for modulo and one for op
        # TODO: Make only one operation

        if op_str in {"mul"}:
            shares = [
                modulo(op(share, y), self.session) // self.fp_encoder.scale
                for share in self.shares
            ]
        else:
            operands_shares = [y] + [0 for _ in range(len(self.shares)-1)]
            shares = [
                modulo(op(*tuple_shares), self.session)
                for tuple_shares in zip(self.shares, operands_shares)
            ]

        result = AdditiveSharingTensor(shares=shares, session=self.session)
        return result

    def __apply_op(self, y, op):
        is_private = isinstance(y, AdditiveSharingTensor)

        if is_private:
            result = self.__apply_private_op(y, op)
        else:
            result = self.__apply_public_op(y, op)

        return result

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t{self.fp_encoder}"

        for share in self.shares:
            out = f"{out}\n\t{share.client} -> {share.__name__}"

        return out

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
