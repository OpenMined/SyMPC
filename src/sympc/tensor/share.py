from sympc.encoder import FixedPointEncoder
from sympc.session import Session
import operator

import torch


class ShareTensor:
    """
    This class represents only 1 share  (from n) that a party
    can generate when secretly sharing that a party holds
    """

    __slots__ = {
        # Populated in Syft
        "id",
        "tags",
        "description",
        "tensor",
        "session",
        "fp_encoder",
    }

    def __init__(
        self,
        data=None,
        session=None,
        encoder_base=2,
        encoder_precision=16,
        ring_size=2 ** 64,
    ):

        if session is None:
            self.session = Session(
                ring_size=ring_size,
            )
            self.session.config.encoder_precision = encoder_precision
            self.session.config.encoder_base = encoder_base

        else:
            self.session = session
            encoder_precision = self.session.config.encoder_precision
            encoder_base = self.session.config.encoder_base

        # TODO: It looks like the same logic as above
        self.fp_encoder = FixedPointEncoder(
            base=encoder_base, precision=encoder_precision
        )

        self.tensor = None
        if data is not None:
            tensor_type = self.session.tensor_type
            self.tensor = self.fp_encoder.encode(data).type(tensor_type)

    @staticmethod
    def sanity_checks(x, y, op_str):
        if op_str == "mul" and isinstance(y, (float, torch.FloatTensor)):
            y = ShareTensor(data=y, session=x.session)
        elif op_str in {"add", "sub"} and not isinstance(y, ShareTensor):
            y = ShareTensor(data=y, session=x.session)

        return y

    def apply_function(self, y, op_str):
        op = getattr(operator, op_str)

        if isinstance(y, ShareTensor):
            value = op(self.tensor, y.tensor)
        else:
            value = op(self.tensor, y)

        res = ShareTensor(session=self.session)
        res.tensor = value
        return res

    def add(self, y):
        y = ShareTensor.sanity_checks(self, y, "add")
        res = self.apply_function(y, "add")
        return res

    def sub(self, y):
        y = ShareTensor.sanity_checks(self, y, "sub")
        res = self.apply_function(y, "sub")
        return res

    def mul(self, y):
        y = ShareTensor.sanity_checks(self, y, "mul")
        res = self.apply_function(y, "mul")

        if isinstance(y, ShareTensor):
            res.tensor = res.tensor // self.fp_encoder.scale

        return res

    def div(self, y):
        # TODO
        pass

    def __getattr__(self, attr_name):
        # Default to some tensor specific attributes like
        # size, shape, etc.
        tensor = self.tensor
        return getattr(tensor, attr_name)

    def __gt__(self, y):
        y = ShareTensor.sanity_checks(self, y, "gt")
        res = self.tensor < y.tensor
        return res

    def __lt__(self, y):
        y = ShareTensor.sanity_checks(self, y, "lt")
        res = self.tensor < y.tensor
        return res

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t| {self.fp_encoder}"
        out = f"{out}\n\t| Data: {self.tensor}"

        return out

    def __eq__(self, other):
        if not (self.tensor == other.tensor).all():
            return False

        if not (self.session == other.session):
            return False

        return True

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = sub
    __mul__ = mul
    __rmul__ = mul
    __div__ = div
