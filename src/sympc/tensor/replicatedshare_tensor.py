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
from sympc.session import Session
from sympc.tensor import ShareTensor
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
        if shares is not None:
            for i in range(len(shares)):
                self.shares.append(self._encode(shares[i]).to(tensor_type))

    def _encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode via FixedPointEncoder.

        Args:
            data (torch.Tensor): Tensor to be encoded

        Returns:
            encoded_data (torch.Tensor): Decoded values

        """
        return self.fp_encoder.encode(data)

    def decode(self) -> List[torch.Tensor]:
        """Decode via FixedPointEncoder.

        Returns:
            List[torch.Tensor]: Decoded values
        """
        return self._decode()

    def _decode(self) -> List[torch.Tensor]:
        """Decodes shares list of RSTensor via FixedPointEncoder.

        Returns:
            List[torch.Tensor]: Decoded values
        """
        shares = []

        for i in range(len(self.shares)):
            tensor = self.fp_encoder.decode(self.shares[i].type(torch.LongTensor))
            shares.append(tensor)

        return shares

    def get_shares(self) -> List[torch.Tensor]:
        """Get shares.

        Returns:
            List[torch.Tensor]: List of shares.
        """
        return self.shares

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

    def eq(self, y: Any):
        """Equal operator.

        Check if "self" is equal with another object given a set of attributes to compare.

        Args:
            y (Any): Object to compare

        Returns:
            bool: True if equal False if not.
        """
        if not (torch.cat(self.shares) == torch.cat(y.shares)).all():
            return False

        if not self.config == y.config:
            return False

        if self.session_uuid and y.session_uuid and self.session_uuid != y.session_uuid:
            return False

        return True

    def __getitem__(self, key: int) -> torch.Tensor:
        """Allows to subset shares.

        Args:
            key (int): The share to be retrieved.

        Returns:
            share (torch.Tensor): Returned share.
        """
        return self.shares[key]

    def __setitem__(self, key: int, newvalue: torch.Tensor) -> None:
        """Allows to set share value to new value.

        Args:
            key (int): The share to be retrieved.
            newvalue (torch.Tensor): New value of share.

        """
        self.shares[key] = newvalue

    def ne(self, y):
        """Not Equal operator.

        Args:
            y: self!=y

        """

    @staticmethod
    def __reconstruct_semi_honest(
        share_ptrs: List["ReplicatedSharedTensor"],
        get_shares: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct value from shares.

        Args:
            share_ptrs (List[RSTensorPointers]): List of RSTensor pointers.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.
        """
        shares1 = share_ptrs[0].get_shares()[0].get()
        shares2 = share_ptrs[1].get_shares().get()

        shares = [shares1] + shares2

        if get_shares:
            return shares

        return sum(shares)

    @staticmethod
    def __reconstruct_malicious(
        share_ptrs: List["ReplicatedSharedTensor"],
        get_shares: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct value from shares.

        Args:
            share_ptrs (List[RSTensorPointers]): List of RSTensor pointers.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.

        Raises:
            ValueError: When parties share values are not equal.
        """
        nparties = len(share_ptrs)

        # Get shares from all parties
        all_shares = []
        for party_rank in range(nparties):
            all_shares.append(share_ptrs[party_rank].get_shares().get())

        # reconstruct shares from all parties and verify
        value = None
        for party_rank in range(nparties):
            share_sum = all_shares[party_rank][0] + sum(
                all_shares[(party_rank + 1) % (nparties)]
            )

            if not value:
                value = share_sum

            elif share_sum != value:
                raise ValueError(
                    "Reconstruction values from all parties are not equal."
                )

        if get_shares:
            return all_shares

        return value

    @staticmethod
    def reconstruct(
        share_ptrs: List["ReplicatedSharedTensor"],
        get_shares: bool = False,
        security_type: str = "semi-honest",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct value from shares.

        Args:
            share_ptrs (List[RSTensorPointers]): List of RSTensor pointers.
            security_type (str): Type of security followed by protocol.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.

        Raises:
            ValueError: Invalid security type
        """
        if security_type == "malicious":
            return ReplicatedSharedTensor.__reconstruct_malicious(
                share_ptrs, get_shares
            )

        elif security_type == "semi-honest":
            return ReplicatedSharedTensor.__reconstruct_semi_honest(
                share_ptrs, get_shares
            )

        raise ValueError("Invalid security Type")

    @staticmethod
    def distribute_shares(shares: List[ShareTensor], session: Session) -> List:
        """Distribute a list of shares.

        Args:
            shares (List[ShareTensor): list of shares to distribute.
            session (Session): Session.

        Returns:
            List of ShareTensorPointers.

        """
        parties = session.parties

        nshares = len(parties) - 1

        ptr_list = []
        for i in range(len(parties)):
            party_ptrs = []

            for j in range(i, i + nshares):
                tensor = shares[j % (nshares + 1)].tensor
                party_ptrs.append(tensor)

            tensor = ReplicatedSharedTensor(
                party_ptrs,
                config=Config(encoder_base=1, encoder_precision=0),
                session_uuid=session.rank_to_uuid[i],
            ).send(parties[i])
            ptr_list.append(tensor)

        return ptr_list

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

            res = ReplicatedSharedTensor(
                session_uuid=_self.session_uuid, config=_self.config
            )
            res.shares = shares
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

            res = ReplicatedSharedTensor(
                session_uuid=_self.session_uuid, config=_self.config
            )
            res.shares = shares
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
    __eq__ = eq
