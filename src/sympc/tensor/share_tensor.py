"""Class used to represent a share owned by a party."""

# stdlib
import operator
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import torch

from sympc.encoder import FixedPointEncoder
from sympc.session import Session

FORWARD_TENSOR_ATTRS = {"unsqueeze", "shape"}


class ShareTensor:
    """This class represents 1 share that a party holds when doing secret sharing.

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
        """Initializer for the ShareTensor.

        Args:
            data (Optional[Any]): The share a party holds. Defaults to None
            session (Optional[Any]): The session from which this shares belongs to.
                Defaults to None.
            encoder_base (int): The base for the encoder. Defaults to 2.
            encoder_precision (int): the precision for the encoder. Defaults to 16.
            ring_size (int): field used for the operations applied on the shares
                Defaults to 2**64
        """
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
        """Decode via FixedPrecisionEncoder.

        Returns:
            torch.Tensor: Decoded value
        """
        return self._decode()

    def _decode(self):
        return self.fp_encoder.decode(self.tensor.type(torch.LongTensor))

    @staticmethod
    def sanity_checks(
        x: "ShareTensor", y: Union[int, float, torch.Tensor, "ShareTensor"], op_str: str
    ) -> "ShareTensor":
        """Check the type of "y" and covert it to share if necessary.

        Args:
            x (ShareTensor): Typically "self".
            y (Union[int, float, torch.Tensor, "ShareTensor"]): Tensor to check.
            op_str (str): String operator.

        Returns:
            ShareTensor: the converted y value.

        """
        if not isinstance(y, ShareTensor):
            y = ShareTensor(data=y, session=x.session)

        return y

    def apply_function(
        self, y: Union["ShareTensor", torch.Tensor, int, float], op_str: str
    ) -> "ShareTensor":
        """Apply a given operation.

        Args:
            y (Union["ShareTensor", torch.Tensor, int, float]): tensor to apply the operator.
            op_str (str): Operator.

        Returns:
            ShareTensor: Result of the operation.
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

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): self + y

        Returns:
            ShareTensor. Result of the operation.
        """
        y_share = ShareTensor.sanity_checks(self, y, "add")
        res = self.apply_function(y_share, "add")
        return res

    def sub(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "sub" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): self - y

        Returns:
            ShareTensor. Result of the operation.
        """
        y_share = ShareTensor.sanity_checks(self, y, "sub")
        res = self.apply_function(y_share, "sub")
        return res

    def rsub(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "sub" operation between "y" and "self".

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): self - y

        Returns:
            ShareTensor. Result of the operation.
        """
        y_share = ShareTensor.sanity_checks(self, y, "sub")
        res = y_share.apply_function(self, "sub")
        return res

    def mul(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): self * y

        Returns:
            ShareTensor. Result of the operation.
        """
        y = ShareTensor.sanity_checks(self, y, "mul")
        res = self.apply_function(y, "mul")

        if self.session.nr_parties == 0:
            # We are using a simple share without usig the MPCTensor
            # In case we used the MPCTensor - the division would have
            # been done in the protocol
            res.tensor = res.tensor // self.fp_encoder.scale

        return res

    def xor(self, y: Union[int, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "xor" operation between "self" and "y".

        Args:
            y (Union[int, torch.Tensor, "ShareTensor"]): self @ y

        Returns:
            ShareTensor: Result of the operation.
        """
        res = ShareTensor(session=self.session)

        if isinstance(y, ShareTensor):
            res.tensor = self.tensor ^ y.tensor
        else:
            res.tensor = self.tensor ^ y

        return res

    def matmul(
        self, y: Union[int, float, torch.Tensor, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): self @ y.

        Returns:
            ShareTensor: Result of the operation.
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
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y (torch.Tensor): y @ self

        Returns:
            ShareTensor. Result of the operation.
        """
        y = ShareTensor.sanity_checks(self, y, "matmul")
        return y.matmul(self)

    def div(self, y: Union[int, float, torch.Tensor, "ShareTensor"]) -> "ShareTensor":
        """Apply the "div" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ShareTensor"]): Denominator.

        Returns:
            ShareTensor: Result of the operation.

        Raises:
            ValueError: If y is not an integer or LongTensor.
        """
        if not isinstance(y, (int, torch.LongTensor)):
            raise ValueError("Div works (for the moment) only with integers!")

        res = ShareTensor(session=self.session)
        res.tensor = self.tensor // y

        return res

    def __getattr__(self, attr_name: str) -> Any:
        """Access to tensor attributes.

        Args:
            attr_name (str): Name of the attribute.

        Returns:
            Any: Attribute.
        """
        tensor = self.tensor
        res = getattr(tensor, attr_name)
        return res

    def __gt__(self, y: Union["ShareTensor", torch.Tensor, int]) -> bool:
        """Greater than operator.

        Args:
            y (Union["ShareTensor", torch.Tensor, int]): Tensor to compare.

        Returns:
            bool: Result of the comparison.
        """
        y_share = ShareTensor.sanity_checks(self, y, "gt")
        res = self.tensor > y_share.tensor
        return res

    def __lt__(self, y: Union["ShareTensor", torch.Tensor, int]) -> bool:
        """Lower than operator.

        Args:
            y (Union["ShareTensor", torch.Tensor, int]): Tensor to compare.

        Returns:
            bool: Result of the comparison.
        """
        y_share = ShareTensor.sanity_checks(self, y, "lt")
        res = self.tensor < y_share.tensor
        return res

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: Return the string representation of ShareTensor.
        """
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t| {self.fp_encoder}"
        out = f"{out}\n\t| Data: {self.tensor}"

        return out

    def __repr__(self) -> str:
        """Representation.

        Returns:
            String representation.
        """
        return self.__str__()

    def __eq__(self, other: Any) -> bool:
        """Equal operator.

        Check if "self" is equal with another object given a set of
            attributes to compare.

        Args:
            other (Any): Tensor to compare.

        Returns:
            bool: True if equal False if not.

        """
        if not (self.tensor == other.tensor).all():
            return False

        if not (self.session == other.session):
            return False

        return True

    # Forward to tensor methods

    @property
    def shape(self) -> Any:
        """Shape of the tensor.

        Returns:
            Any: Shape of the tensor.
        """
        return self.tensor.shape

    def numel(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        """Number of elements.

        Args:
            *args: Arguments passed to tensor.numel.
            **kwargs: Keyword arguments passed to tensor.numel.

        Returns:
            Any: Total number of elements of the tensor.

        """
        return self.tensor.numel(*args, **kwargs)

    @property
    def T(self) -> Any:
        """Transpose.

        Returns:
            Any: ShareTensor transposed.

        """
        res = ShareTensor(session=self.session)
        res.tensor = self.tensor.T
        return res

    def unsqueeze(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        """New tensor with a dimension of size one inserted at the specified position.

        Args:
            *args: Arguments to tensor.unsqueeze
            **kwargs: Keyword arguments passed to tensor.unsqueeze

        Returns:
            Any: Tensor unsqueezed.

        References:
            https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        """
        tensor = self.tensor.unsqueeze(*args, **kwargs)
        res = ShareTensor(session=self.session)
        res.tensor = tensor
        return res

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = div
    __xor__ = xor
