"""Used to abstract multiple shared values held by parties."""

# stdlib
import operator
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

import sympc
from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.tensor import ShareTensor
from sympc.utils import get_type_from_ring
from sympc.utils import islocal
from sympc.utils import parallel_execution

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
    PROPERTIES_FORWARD = {"T", "shape"}

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
            self.shares = [self._encode(share).to(tensor_type) for share in shares]

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

        shares = [
            self.fp_encoder.decode(share.type(torch.LongTensor))
            for share in self.shares
        ]
        return shares

    def get_shares(self) -> List[torch.Tensor]:
        """Get shares.

        Returns:
            List[torch.Tensor]: List of shares.
        """
        return self.shares

    @staticmethod
    def sanity_checks(
        x: "ReplicatedSharedTensor",
        y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"],
    ) -> "ReplicatedSharedTensor":
        """Check the type of "y" and convert it to share if necessary.

        Args:
            x (ReplicatedSharedTensor): Typically "self".
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): Tensor to check.

        Returns:
            ReplicatedSharedTensor: the converted y value.

        Raises:
            ValueError: if both values are shares and they have different uuids
            ValueError: if both values have different number of shares.
        """
        if not isinstance(y, ReplicatedSharedTensor):
            if x.session_uuid is not None:
                session = sympc.session.get_session(str(x.session_uuid))
                ring_size = session.ring_size
                config = session.config
            else:
                ring_size = x.ring_size
                config = x.config

            y = ReplicatedSharedTensor(shares=[y], ring_size=ring_size, config=config)

        elif y.session_uuid and x.session_uuid and y.session_uuid != x.session_uuid:
            raise ValueError(
                f"Session UUIDs did not match {x.session_uuid} {y.session_uuid}"
            )
        elif len(x.shares) != len(y.shares):
            raise ValueError("Both RSTensors should have equal number of shares.")

        session_uuid = x.session_uuid

        if session_uuid is not None:
            session = sympc.session.get_session(str(x.session_uuid))
        else:
            ring_size = x.ring_size
            config = x.config
            session = Session(config=config, ring_size=ring_size)
            session.nr_parties = 1

        return y, session

    def __apply_public_op(
        self, y: Union[torch.Tensor, float, int], op_str: str
    ) -> "ReplicatedSharedTensor":
        """Apply an operation on "self" which is a RSTensor and a public value.

        Args:
            y (Union[torch.Tensor, float, int]): Tensor to apply the operation.
            op_str (str): The operation.

        Returns:
            ReplicatedSharedTensor: The operation "op_str" applied on "self" and "y"

        Raises:
            ValueError: If "op_str" is not supported.
        """
        y, session = ReplicatedSharedTensor.sanity_checks(self, y)
        session_uuid = self.session_uuid

        op = getattr(operator, op_str)
        shares = self.shares
        if op_str in {"add", "sub"}:
            if session.rank != 1:
                idx = (session.nr_parties - session.rank) % session.nr_parties
                shares[idx] = op(shares[idx], y.shares[0])
        else:
            raise ValueError(f"{op_str} not supported")

        result = ReplicatedSharedTensor(
            ring_size=session.ring_size,
            session_uuid=session_uuid,
            config=session.config,
        )
        result.shares = shares
        return result

    def __apply_private_op(
        self, y: "ReplicatedSharedTensor", op_str: str
    ) -> "ReplicatedSharedTensor":
        """Apply an operation on 2 RSTensors (secret shared values).

        Args:
            y (RSTensor): Tensor to apply the operation
            op_str (str): The operation

        Returns:
            ReplicatedSharedTensor: The operation "op_str" applied on "self" and "y"

        Raises:
            ValueError: If "op_str" not supported.
        """
        y, session = ReplicatedSharedTensor.sanity_checks(self, y)

        op = getattr(operator, op_str)
        shares = []
        if op_str in {"add", "sub"}:
            for x_share, y_share in zip(self.shares, y.shares):
                shares.append(op(x_share, y_share))
        else:
            raise ValueError(f"{op_str} not supported")

        result = ReplicatedSharedTensor(
            ring_size=session.ring_size,
            session_uuid=self.session_uuid,
            config=session.config,
        )
        result.shares = shares
        return result

    def __apply_op(
        self,
        y: Union["ReplicatedSharedTensor", torch.Tensor, float, int],
        op_str: str,
    ) -> "ReplicatedSharedTensor":
        """Apply a given operation ".

         This function checks if "y" is private or public value.

        Args:
            y: tensor to apply the operation.
            op_str: the operation.

        Returns:
            ReplicatedSharedTensor: the operation "op_str" applied on "self" and "y"
        """
        is_private = isinstance(y, ReplicatedSharedTensor)

        if is_private:
            result = self.__apply_private_op(y, op_str)
        else:
            result = self.__apply_public_op(y, op_str)

        return result

    def add(
        self, y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): self + y

        Returns:
            ReplicatedSharedTensor: Result of the operation.
        """
        return self.__apply_op(y, "add")

    def sub(
        self, y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "sub" operation between  "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): self - y

        Returns:
            ReplicatedSharedTensor: Result of the operation.
        """
        return self.__apply_op(y, "sub")

    def rsub(
        self, y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "sub" operation between "y" and "self".

        Args:
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): y -self

        Returns:
            ReplicatedSharedTensor: Result of the operation.
        """
        return self.__apply_op(y, "sub")

    def mul(self, y: Union[int, float, torch.Tensor]) -> "ReplicatedSharedTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y: self*y

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: Raised when private mul is performed parties!=3.

        """
        y_tensor, session = self.sanity_checks(self, y)
        is_private = isinstance(y, ReplicatedSharedTensor)

        if is_private:
            if session.nr_parties == 3:
                from sympc.protocol import Falcon

                result = [Falcon.multiplication_protocol(self, y_tensor)]
            else:
                raise ValueError(
                    "Private mult between ReplicatedSharedTensors is allowed only for 3 parties"
                )
        else:
            result = [share * y_tensor.shares[0] for share in self.shares]

        tensor = ReplicatedSharedTensor(
            ring_size=self.ring_size, session_uuid=self.session_uuid, config=self.config
        )
        tensor.shares = result

        return tensor

    def truediv(self, y: Union[int, torch.Tensor]) -> "ReplicatedSharedTensor":
        """Apply the "div" operation between "self" and "y".

        Args:
            y (Union[int , torch.Tensor]): Denominator.

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: If y is not an integer or LongTensor.
        """
        if not isinstance(y, (int, torch.LongTensor)):
            raise ValueError(
                "Div works (for the moment) only with integers and LongTensor!"
            )

        res = ReplicatedSharedTensor(session_uuid=self.session_uuid, config=self.config)
        res.shares = [share // y for share in self.shares]
        return res

    def rshift(self, y: int) -> "ReplicatedSharedTensor":
        """Apply the "rshift" operation to "self".

        Args:
            y (int): shift value

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: If y is not an integer.
        """
        if not isinstance(y, int):
            raise ValueError("Right Shift works only with integers!")

        res = ReplicatedSharedTensor(session_uuid=self.session_uuid, config=self.config)
        res.shares = [share >> y for share in self.shares]
        return res

    def matmul(self, y):
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y: self@y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def rmatmul(self, y):
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y: self@y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def xor(self, y):
        """Apply the "xor" operation between "self" and "y".

        Args:
            y: self^y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def lt(self, y):
        """Lower than operator.

        Args:
            y: self<y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def gt(self, y):
        """Greater than operator.

        Args:
            y: self>y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def eq(self, y: Any) -> bool:
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

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    @staticmethod
    def _request_and_get(
        share_ptr: "ReplicatedSharedTensor",
    ) -> "ReplicatedSharedTensor":
        """Function used to request and get a share - Duet Setup.

        Args:
            share_ptr (ReplicatedSharedTensor): input ReplicatedSharedTensor

        Returns:
            ReplicatedSharedTensor : The ReplicatedSharedTensor in local.
        """
        if not islocal(share_ptr):
            share_ptr.request(block=True)
        res = share_ptr.get_copy()
        return res

    @staticmethod
    def __reconstruct_semi_honest(
        share_ptrs: List["ReplicatedSharedTensor"],
        get_shares: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct value from shares.

        Args:
            share_ptrs (List[ReplicatedSharedTensor]): List of RSTensor pointers.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.
        """
        request = ReplicatedSharedTensor._request_and_get
        request_wrap = parallel_execution(request)
        args = [[share] for share in share_ptrs[:2]]
        local_shares = request_wrap(args)

        shares = [local_shares[0].shares[0]]
        shares.extend(local_shares[1].shares)

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
            share_ptrs (List[ReplicatedSharedTensor]): List of RSTensor pointers.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.

        Raises:
            ValueError: When parties share values are not equal.
        """
        nparties = len(share_ptrs)

        # Get shares from all parties
        request = ReplicatedSharedTensor._request_and_get
        request_wrap = parallel_execution(request)
        args = [[share] for share in share_ptrs]
        local_shares = request_wrap(args)

        all_shares = [rst.shares for rst in local_shares]
        # reconstruct shares from all parties and verify
        value = None
        for party_rank in range(nparties):
            share_sum = all_shares[party_rank][0] + sum(
                all_shares[(party_rank + 1) % (nparties)]
            )

            if value is None:
                value = share_sum
            elif (share_sum != value).any():
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
            share_ptrs (List[ReplicatedSharedTensor]): List of RSTensor pointers.
            security_type (str): Type of security followed by protocol.
            get_shares (bool): Retrieve only shares.

        Returns:
            reconstructed_value (torch.Tensor): Reconstructed value.

        Raises:
            ValueError: Invalid security type.
            ValueError : SharePointers not provided.
        """
        if not len(share_ptrs):
            raise ValueError("Share pointers must be provided for reconstruction.")
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
    def distribute_shares_to_party(
        shares: List[Union[ShareTensor, torch.Tensor]],
        party_rank: int,
        session: Session,
    ) -> "ReplicatedSharedTensor":
        """Distributes shares to party.

        Args:
            shares (List[Union[ShareTensor,torch.Tensor]]): Shares to be distributed.
            party_rank (int): Rank of party.
            session (Session): Current session

        Returns:
            tensor (ReplicatedSharedTensor): Tensor with shares

        Raises:
            TypeError: Invalid share class.
        """
        party = session.parties[party_rank]
        nshares = session.nr_parties - 1
        party_shares = []

        for share_index in range(party_rank, party_rank + nshares):
            share = shares[share_index % (nshares + 1)]

            if isinstance(share, torch.Tensor):
                party_shares.append(share)

            elif isinstance(share, ShareTensor):
                party_shares.append(share.tensor)

            else:
                raise TypeError(f"{type(share)} is an invalid share class")

        tensor = ReplicatedSharedTensor(
            party_shares,
            config=Config(encoder_base=1, encoder_precision=0),
            session_uuid=session.rank_to_uuid[party_rank],
        ).send(party)

        return tensor

    @staticmethod
    def distribute_shares(
        shares: List[Union[ShareTensor, torch.Tensor]], session: Session
    ) -> List["ReplicatedSharedTensor"]:
        """Distribute a list of shares.

        Args:
            shares (List[ShareTensor): list of shares to distribute.
            session (Session): Session.

        Returns:
            List of ReplicatedShareTensors.

        Raises:
            TypeError: when Datatype of shares is invalid.

        """
        if not isinstance(shares, list):
            raise TypeError("Shares to be distributed should be a list of shares")

        if len(shares) != session.nr_parties:
            return ValueError(
                "Number of shares to be distributed should be same as number of parties"
            )
        args = [
            [shares, party_rank, session] for party_rank in range(session.nr_parties)
        ]

        return [ReplicatedSharedTensor.distribute_shares_to_party(*arg) for arg in args]

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

            for share in _self.shares:
                tensor = getattr(share, property_name)
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
            for share in _self.shares:
                tensor = getattr(share, method_name)(*args, **kwargs)
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
    __floordiv__ = truediv
    __xor__ = xor
    __eq__ = eq
    __rshift__ = rshift
