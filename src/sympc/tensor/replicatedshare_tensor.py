"""Used to abstract multiple shared values held by parties."""

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from uuid import UUID

# third party
import torch

from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.utils import get_type_from_ring

from .tensor import SyMPCTensor

PROPERTIES_NEW_RS_TENSOR: Set[str] = {"T"}
METHODS_NEW_RS_TENSOR: Set[str] = {"unsqueeze", "view", "t", "sum", "clone"}


class ReplicatedSharedTensor(metaclass=SyMPCTensor):
    """RSTensor is used when a party holds more than a single share,required by various protocols.

    Arguments:
       shares (Optional[List[Union[float, int, torch.Tensor]]]): Shares list
           from which RSTensor is created.


    Attributes:
       shares: The shares held by the party
    """

    AUTOGRAD_IS_ON: bool = True

    # Used by the SyMPCTensor metaclass
    METHODS_FORWARD = {"numel", "t", "unsqueeze", "view", "sum", "clone"}
    PROPERTIES_FORWARD = {"T"}

    def __init__(
        self,
        shares: Optional[List[Union[float, int, torch.Tensor]]] = None,
        config: Config = Config(encoder_base=2, encoder_precision=16),
        session_uuid: Optional[UUID] = None,
        ring_size: int = 2 ** 64,
    ):
        """Initialize ReplicatedSharedTensor.

        Args:
            shares (Optional[List[Union[float, int, torch.Tensor]]]): Shares list
                from which RSTensor is created.
            config (Config): The configuration where we keep the encoder precision and base.
            session_uuid (Optional[UUID]): Used to keep track of a share that is associated with a
                remote session
            ring_size (int): field used for the operations applied on the shares
                Defaults to 2**64

        """
        self.session_uuid = session_uuid
        self.ring_size = ring_size

        self.config = config
        self.fp_encoder = FixedPointEncoder(
            base=config.encoder_base, precision=config.encoder_precision
        )

        tensor_type = get_type_from_ring(ring_size)
        self.shares = []
        for i in range(len(shares)):
            self.shares.append(self._encode(shares[i]).to(tensor_type))

    def _encode(self, data):
        return self.fp_encoder.encode(data)

    def decode(self):
        """Decode via FixedPrecisionEncoder.

        Returns:
            torch.Tensor: Decoded value
        """
        return self._decode()

    def _decode(self):

        shares = []
        for i in range(len(self.shares)):
            tensor = self.fp_encoder.decode(self.shares[i].type(torch.LongTensor))
            shares.append(tensor)

        return shares

    def add(self, y):
        """Apply the "add" operation between "self" and "y".

        Args:
            y: self+y

        """

    def sub(self, y):
        """Apply the "sub" operation between "self" and "y".

        Args:
            y: self-y


        """

    def rsub(self, y):
        """Apply the "sub" operation between "y" and "self".

        Args:
            y: self-y

        """

    def mul(self, y):
        """Apply the "mul" operation between "self" and "y".

        Args:
            y: self*y

        """

    def truediv(self, y):
        """Apply the "div" operation between "self" and "y".

        Args:
            y: self/y

        """

    def matmul(self, y):
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y: self@y

        """

    def rmatmul(self, y):
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y: self@y

        """

    def xor(self, y):
        """Apply the "xor" operation between "self" and "y".

        Args:
            y: self^y

        """

    def lt(self, y):
        """Lower than operator.

        Args:
            y: self<y

        """

    def gt(self, y):
        """Greater than operator.

        Args:
            y: self>y

        """

    def eq(self, y):
        """Equal operator.

        Args:
            y: self==y

        """

    def ne(self, y):
        """Not Equal operator.

        Args:
            y: self!=y

        """

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

        def property_new_rs_tensor_getter(_self: "ReplicatedSharedTensor") -> Any:
            shares = []

            for i in range(len(_self.shares)):
                tensor = getattr(_self.shares[i], property_name)
                shares.append(tensor)

            res = ReplicatedSharedTensor(session=_self.session, shares=shares)

            return res

        def property_getter(_self: "ReplicatedSharedTensor") -> Any:
            prop = getattr(_self.shares[0], property_name)
            return prop

        if property_name in PROPERTIES_NEW_RS_TENSOR:
            res = property(property_new_rs_tensor_getter, None)
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

        def method_new_rs_tensor(
            _self: "ReplicatedSharedTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            shares = []
            for i in range(len(_self.shares)):
                tensor = getattr(_self.shares[i], method_name)(*args, **kwargs)
                shares.append(tensor)

            res = ReplicatedSharedTensor(session=_self.session, shares=shares)

            return res

        def method(
            _self: "ReplicatedSharedTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.shares[0], method_name)
            res = method(*args, **kwargs)
            return res

        if method_name in METHODS_NEW_RS_TENSOR:
            res = method_new_rs_tensor
        else:
            res = method

        return res

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = truediv
    __xor__ = xor
