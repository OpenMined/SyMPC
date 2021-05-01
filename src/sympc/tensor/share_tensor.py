"""Class used to represent a share owned by a party."""

# stdlib
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union
import functools

# third party
import torch

from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.tensor.grads import forward
from sympc.tensor.grads import GRAD_FUNCS

from .tensor import SyMPCTensor

PROPERTIES_NEW_SHARE_TENSOR: Set[str] = {"T"}
METHODS_NEW_SHARE_TENSOR: Set[str] = {"unsqueeze", "view", "t"}


def wrapper_getattribute(func):
    def wrapper_func(*args, **kwargs):
        _self, *new_args = args
        f = getattr(_self, func.__name__)
        return f(*new_args, **kwargs)

    return wrapper_func


class ShareTensor(metaclass=SyMPCTensor):
    """Single Share representation.

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
        # We need this because only floating type tensor can have requires_grad
        # If not, we could use the self.tensor requires_grad
        "requires_grad",
        # Use for training
        "grad",
        "grad_fn",
        "ctx",
        "parents",
    }

    AUTOGRAD_IS_ON: bool = True

    # Used by the SyMPCTensor metaclass
    METHODS_FORWARD: Set[str] = {"numel", "unsqueeze", "t", "view"}
    PROPERTIES_FORWARD: Set[str] = {"T", "shape"}

    def __init__(
        self,
        data: Optional[Union[float, int, torch.Tensor]] = None,
        session: Optional[Session] = None,
        encoder_base: int = 2,
        encoder_precision: int = 16,
        ring_size: int = 2 ** 64,
        requires_grad: bool = False,
    ) -> None:
        """Initialize ShareTensor.

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

        self.grad_fn = None
        self.grad = 0
        self.ctx = {}
        self.requires_grad = requires_grad
        self.parents = []

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
            y (Union[int, float, torch.Tensor, "ShareTensor"]): y - self

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
            y (Union[int, torch.Tensor, "ShareTensor"]): self xor y

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

    def __getattribute__(self, attr_name: str) -> Any:
        # Do the forward pass
        # Implementation similar to CrypTen
        grad_fn = GRAD_FUNCS.get(attr_name, None)
        if grad_fn and ShareTensor.AUTOGRAD_IS_ON:
            return functools.partial(forward, self, grad_fn)

        return object.__getattribute__(self, attr_name)

    def backward(self) -> Any:
        # TODO: implement this
        pass

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
        """Representation.

        Returns:
            str: Return the string representation of ShareTensor.
        """
        type_name = type(self).__name__
        out = f"[{type_name}]"
        out = f"{out}\n\t| {self.fp_encoder}"
        out = f"{out}\n\t| Data: {self.tensor}"

        if self.grad_fn:
            out = f"{out}\n\t| GradFunc: {self.grad_fn}"

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

    @staticmethod
    def hook_property(property_name: str) -> Any:
        """Hook a framework property (only getter).

        Ex:
         * if we call "shape" we want to call it on the underlying tensor
        and return the result
         * if we call "T" we want to call it on the underlying tensor
        but we want to wrap it in the same tensor type

        Args:
            property_name (str): property to hook

        Returns:
            A hooked property
        """

        def property_new_share_tensor_getter(_self: "ShareTensor") -> Any:
            tensor = getattr(_self.tensor, property_name)
            res = ShareTensor(session=_self.session)
            res.tensor = tensor
            return res

        def property_getter(_self: "ShareTensor") -> Any:
            prop = getattr(_self.tensor, property_name)
            return prop

        if property_name in PROPERTIES_NEW_SHARE_TENSOR:
            res = property(property_new_share_tensor_getter, None)
        else:
            res = property(property_getter, None)

        return res

    @staticmethod
    def hook_method(method_name: str) -> Callable[..., Any]:
        """Hook a framework method such that we know how to treat it given that we call it.

        Ex:
         * if we call "numel" we want to call it on the underlying tensor
        and return the result
         * if we call "unsqueeze" we want to call it on the underlying tensor
        but we want to wrap it in the same tensor type

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_new_share_tensor(
            _self: "ShareTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.tensor, method_name)
            tensor = method(*args, **kwargs)
            res = ShareTensor(session=_self.session)
            res.tensor = tensor
            return res

        def method(
            _self: "ShareTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.tensor, method_name)
            res = method(*args, **kwargs)
            return res

        if method_name in METHODS_NEW_SHARE_TENSOR:
            res = method_new_share_tensor
        else:
            res = method

        return res

    __add__ = wrapper_getattribute(add)
    __radd__ = wrapper_getattribute(add)
    __sub__ = wrapper_getattribute(sub)
    __rsub__ = wrapper_getattribute(rsub)
    __mul__ = wrapper_getattribute(mul)
    __rmul__ = wrapper_getattribute(mul)
    __matmul__ = wrapper_getattribute(matmul)
    __rmatmul__ = wrapper_getattribute(rmatmul)
    __truediv__ = wrapper_getattribute(div)
    __xor__ = wrapper_getattribute(xor)
