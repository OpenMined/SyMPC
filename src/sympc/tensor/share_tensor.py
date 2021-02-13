"""Class used to represent a share owned by a party."""

# stdlib
import operator
from typing import Any
from typing import Optional
from typing import Union

# third party
import torch

from sympc.encoder import FixedPointEncoder
from sympc.session import Session


tensor_methods = {"unsqueeze", "shape"}


class SYMPCTensor(type):
    def __getattribute__(cls, name):
        if name in tensor_methods:
            return None
        return super().__getattribute__(name)


class ShareTensor(metaclass=SYMPCTensor):
    """This class represents 1 share that a party holds when doing secret
    sharing.

    Arguments:
        data (Optional[Any]): the share a party holds
        session (Optional[Any]): the session from which this shares belongs to
        encoder_base (int): the base for the encoder
        encoder_precision (int): the precision for the encoder
        ring_size (int): field used for the operations applied on the shares

    Attributes:
        Syft Serializable Attributes

        id (UID): the id to store the session
        tags (Optional[List[str]): an optional list of strings that are tags used at search
        description (Optional[str]): an optional string used to describe the session

        tensor (Any): the value of the share
        session (Session): keep track from which session  this share belongs to
        fp_encoder (FixedPointEncoder): the encoder used to convert a share from/to fixed point
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
        data: Optional[Union[float, int, torch.Tensor]] = None,
        session: Optional[Session] = None,
        encoder_base: int = 2,
        encoder_precision: int = 16,
        ring_size: int = 2 ** 64,
    ) -> None:
        """Initializer for the ShareTensor."""

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

        self.tensor: Optional[torch.Tensor] = None

        if data is not None:
            tensor_type = self.session.tensor_type
            self.tensor = self._encode(data).type(tensor_type)

    def _encode(self, data):
        return self.fp_encoder.encode(data)

    def decode(self):
        return self._decode()

    def _decode(self):
        return self.fp_encoder.decode(self.tensor.type(torch.LongTensor))

    @staticmethod
    def sanity_checks(
        x: "ShareTensor", y: Union[int, float, torch.Tensor, "ShareTensor"], op_str: str
    ) -> "ShareTensor":
        """Check the type of "y" and convert it to a share if necessary.

        :return: the y value
        :rtype: ShareTensor, int or Integer type Tensor
        """
        if not isinstance(y, ShareTensor):
            y = ShareTensor(data=y, session=x.session)

        return y

    def apply_function(
        self, y: Union["ShareTensor", torch.Tensor, int, float], op_str: str
    ) -> "ShareTensor":
        """Apply a given operation.

        :return: the result of applying "op_str" on "self" and y
        :rtype: ShareTensor
        """
        op = getattr(operator, op_str)

        if isinstance(y, ShareTensor):
            value = op(self.tensor, y.tensor)
        else:
            value = op(self.tensor, y)

        res = ShareTensor(session=self.session)
        res.tensor = value
        return res

    def add(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "add" operation between "self" and "y".

        :return: self + y
        :rtype: ShareTensor
        """
        y_share = ShareTensor.sanity_checks(self, y, "add")
        res = self.apply_function(y_share, "add")
        return res

    def sub(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "sub" operation between "self" and "y".

        :return: self - y
        :rtype: ShareTensor
        """
        y_share = ShareTensor.sanity_checks(self, y, "sub")
        res = self.apply_function(y_share, "sub")
        return res

    def rsub(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "sub" operation between "y" and "self".

        :return: y - self
        :rtype: ShareTensor
        """
        y_share = ShareTensor.sanity_checks(self, y, "sub")
        res = y_share.apply_function(self, "sub")
        return res

    def mul(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "mul" operation between "self" and "y".

        :return: self * y
        :rtype: ShareTensor
        """
        y = ShareTensor.sanity_checks(self, y, "mul")
        res = self.apply_function(y, "mul")

        if self.session.nr_parties == 0:
            # We are using a simple share without usig the MPCTensor
            # In case we used the MPCTensor - the division would have
            # been done in the protocol
            res.tensor = res.tensor // self.fp_encoder.scale

        return res

    def matmul(
        self, y: Union[int, float, torch.Tensor, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "matmul" operation between "self" and "y".

        :return: self @ y
        :rtype: ShareTensor
        """
        y = ShareTensor.sanity_checks(self, y, "matmul")
        res = self.apply_function(y, "matmul")

        if self.session.nr_parties == 0:
            # We are using a simple share without usig the MPCTensor
            # In case we used the MPCTensor - the division would have
            # been done in the protocol
            res.tensor = res.tensor // self.fp_encoder.scale

        return res

    def rmatmul(self, y: torch.Tensor) -> "ShareTensor":
        """Apply the reversed "matmul" operation between "self" and "y".

        :return: y @ self
        :rtype: ShareTensor
        """
        y = ShareTensor.sanity_checks(self, y, "matmul")
        return y.matmul(self)

    def div(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "div" operation between "self" and "y". Currently,
        NotImplemented.

        :return: self / y
        :rtype: ShareTensor
        """
        if not isinstance(y, (int, torch.LongTensor)):
            raise ValueError("Div works (for the moment) only with integers!")

        res = ShareTensor(session=self.session)
        res.tensor = self.tensor // y

        return res

    def __getattribute__(self, attr_name: str) -> Any:
        attr = super().__getattribute__(attr_name)
        return attr

    def __getattr__(self, attr_name: str) -> Any:
        """Get the attribute from the ShareTensor. If the attribute is not
        found at the ShareTensor level, the it would look for in the the
        "tensor".

        :return: the attribute value
        :rtype: Anything
        """
        # Default to some tensor specific attributes like
        # size, shape, etc.
        tensor = self.tensor
        return getattr(tensor, attr_name)

    def __gt__(self, y: Union["ShareTensor", torch.Tensor, int]) -> bool:
        """Check if "self" is bigger than "y".

        :return: self > y
        :rtype: bool
        """
        y_share = ShareTensor.sanity_checks(self, y, "gt")
        res = self.tensor > y_share.tensor
        return res

    def __lt__(self, y: Union["ShareTensor", torch.Tensor, int]) -> bool:
        """Check if "self" is less than "y".

        :return: self < y
        :rtype: bool
        """

        y_share = ShareTensor.sanity_checks(self, y, "lt")
        res = self.tensor < y_share.tensor
        return res

    def __str__(self) -> str:
        """Return the string representation of ShareTensor."""
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t| {self.fp_encoder}"
        out = f"{out}\n\t| Data: {self.tensor}"

        return out

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of
        attributes to compare.

        :return: if self and other are equal
        :rtype: bool
        """

        if not (self.tensor == other.tensor).all():
            return False

        if not (self.session == other.session):
            return False

        return True

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = div


