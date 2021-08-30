"""Used to abstract multiple shared values held by parties."""

# stdlib
import copy
import dataclasses
from functools import reduce
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
from sympc.utils import RING_SIZE_TO_TYPE
from sympc.utils import get_nr_bits
from sympc.utils import get_type_from_ring
from sympc.utils import islocal
from sympc.utils import ispointer
from sympc.utils import parallel_execution

from .tensor import SyMPCTensor

PROPERTIES_NEW_RS_TENSOR: Set[str] = {"T"}
METHODS_NEW_RS_TENSOR: Set[str] = {"unsqueeze", "view", "t", "sum", "clone", "repeat"}
BINARY_MAP = {"add": "xor", "sub": "xor", "mul": "and_"}

PRIME_NUMBER = 67  # Global constant for prime order rings.


class ReplicatedSharedTensor(metaclass=SyMPCTensor):
    """RSTensor is used when a party holds more than a single share,required by various protocols.

    Arguments:
       shares (Optional[List[Union[float, int, torch.Tensor]]]): Shares list
           from which RSTensor is created.


    Attributes:
       shares: The shares held by the party
    """

    __slots__ = {
        # Populated in Syft
        "id",
        "tags",
        "description",
        "shares",
        "session_uuid",
        "config",
        "fp_encoder",
        "ring_size",
    }

    # Used by the SyMPCTensor metaclass
    METHODS_FORWARD = {"numel", "t", "unsqueeze", "view", "sum", "clone", "repeat"}
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

        if ring_size in {2, PRIME_NUMBER}:
            self.config = Config(encoder_base=1, encoder_precision=0)
        else:
            self.config = config

        self.fp_encoder = FixedPointEncoder(
            base=self.config.encoder_base, precision=self.config.encoder_precision
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

    def get_ring_size(self) -> str:
        """Ring size of tensor.

        Returns:
            ring_size (str): Returns ring_size of tensor in string.

        It is typecasted to string as we cannot serialize 2**64
        """
        return str(self.ring_size)

    def get_config(self) -> Dict:
        """Config of tensor.

        Returns:
            config (Dict): returns config of the tensor as dict.
        """
        return dataclasses.asdict(self.config)

    @staticmethod
    def addmodprime(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes addition(x+y) modulo PRIME_NUMBER constant.

        Args:
            x (torch.Tensor): input tensor
            y (torch.tensor): input tensor

        Returns:
            value (torch.Tensor): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not torch.uint8
        """
        if x.dtype != torch.uint8 or y.dtype != torch.uint8:
            raise ValueError(
                f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
            )

        return (x + y) % PRIME_NUMBER

    @staticmethod
    def submodprime(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes subtraction(x-y) modulo PRIME_NUMBER constant.

        Args:
            x (torch.Tensor): input tensor
            y (torch.tensor): input tensor

        Returns:
            value (torch.Tensor): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not torch.uint8
        """
        if x.dtype != torch.uint8 or y.dtype != torch.uint8:
            raise ValueError(
                f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
            )

        # Typecasting is done, as underflow returns a positive number,as it is unsigned.
        x = x.to(torch.int8)
        y = y.to(torch.int8)

        result = (x - y) % PRIME_NUMBER

        return result.to(torch.uint8)

    @staticmethod
    def mulmodprime(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes multiplication(x*y) modulo PRIME_NUMBER constant.

        Args:
            x (torch.Tensor): input tensor
            y (torch.tensor): input tensor

        Returns:
            value (torch.Tensor): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not torch.uint8
        """
        if x.dtype != torch.uint8 or y.dtype != torch.uint8:
            raise ValueError(
                f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
            )

        # We typecast as multiplication result in 2n bits ,which causes overflow.
        x = x.to(torch.int16)
        y = y.to(torch.int16)

        result = (x * y) % PRIME_NUMBER

        return result.to(torch.uint8)

    @staticmethod
    def get_op(ring_size: int, op_str: str) -> Callable[..., Any]:
        """Returns method attribute based on ring_size and op_str.

        Args:
            ring_size (int): Ring size
            op_str (str): Operation string.

        Returns:
            op (Callable[...,Any]): The operation method for the op_str.

        Raises:
            ValueError : If invalid ring size is given as input.
        """
        op = None
        if ring_size == 2:
            op = getattr(operator, BINARY_MAP[op_str])
        elif ring_size == PRIME_NUMBER:
            op = getattr(ReplicatedSharedTensor, op_str + "modprime")
        elif ring_size in RING_SIZE_TO_TYPE.keys():
            op = getattr(operator, op_str)
        else:
            raise ValueError(f"Invalid ring size: {ring_size}")

        return op

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
            ValueError: if both RSTensor have different ring_sizes
        """
        if not isinstance(y, ReplicatedSharedTensor):
            # As prime ring size is unsigned,we convert negative values.
            y = y % PRIME_NUMBER if x.ring_size == PRIME_NUMBER else y

            y = ReplicatedSharedTensor(
                session_uuid=x.session_uuid,
                shares=[y],
                ring_size=x.ring_size,
                config=x.config,
            )

        elif y.session_uuid and x.session_uuid and y.session_uuid != x.session_uuid:
            raise ValueError(
                f"Session UUIDs did not match {x.session_uuid} {y.session_uuid}"
            )
        elif len(x.shares) != len(y.shares):
            raise ValueError(
                f"Both RSTensors should have equal number of shares {len(x.shares)} {len(y.shares)}"
            )
        elif x.ring_size != y.ring_size:
            raise ValueError(
                f"Both RSTensors should have same ring_size {x.ring_size} {y.ring_size}"
            )

        session_uuid = x.session_uuid

        if session_uuid is not None:
            session = sympc.session.get_session(str(x.session_uuid))
        else:
            session = Session(config=x.config, ring_size=x.ring_size)
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

        op = ReplicatedSharedTensor.get_op(self.ring_size, op_str)

        shares = copy.deepcopy(self.shares)
        if op_str in {"add", "sub"}:
            if session.rank != 1:
                idx = (session.nr_parties - session.rank) % session.nr_parties
                shares[idx] = op(shares[idx], y.shares[0])
        else:
            raise ValueError(f"{op_str} not supported")

        result = ReplicatedSharedTensor(
            ring_size=self.ring_size,
            session_uuid=self.session_uuid,
            config=self.config,
        )
        result.shares = shares
        return result

    def __apply_private_op(
        self, y: "ReplicatedSharedTensor", op_str: str
    ) -> "ReplicatedSharedTensor":
        """Apply an operation on 2 RSTensors (secret shared values).

        Args:
            y (RelicatedSharedTensor): Tensor to apply the operation
            op_str (str): The operation

        Returns:
            ReplicatedSharedTensor: The operation "op_str" applied on "self" and "y"

        Raises:
            ValueError: If "op_str" not supported.
        """
        y, session = ReplicatedSharedTensor.sanity_checks(self, y)

        op = ReplicatedSharedTensor.get_op(self.ring_size, op_str)

        shares = []
        if op_str in {"add", "sub"}:
            for x_share, y_share in zip(self.shares, y.shares):
                shares.append(op(x_share, y_share))
        else:
            raise ValueError(f"{op_str} not supported")

        result = ReplicatedSharedTensor(
            ring_size=self.ring_size,
            session_uuid=self.session_uuid,
            config=self.config,
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
            y (Union[ReplicatedSharedTensor,torch.Tensor, float, int]): tensor
                to apply the operation.
            op_str (str): the operation.

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

    def mul(
        self, y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): self*y

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: Raised when private mul is performed parties!=3.

        """
        y_tensor, session = self.sanity_checks(self, y)
        is_private = isinstance(y, ReplicatedSharedTensor)

        op_str = "mul"
        op = ReplicatedSharedTensor.get_op(self.ring_size, op_str)
        if is_private:
            if session.nr_parties == 3:
                from sympc.protocol import Falcon

                result = [Falcon.multiplication_protocol(self, y_tensor, op_str)]
            else:
                raise ValueError(
                    "Private mult between ReplicatedSharedTensors is allowed only for 3 parties"
                )
        else:
            result = [op(share, y_tensor.shares[0]) for share in self.shares]

        tensor = ReplicatedSharedTensor(
            ring_size=self.ring_size, session_uuid=self.session_uuid, config=self.config
        )
        tensor.shares = result

        return tensor

    def matmul(
        self, y: Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, "ReplicatedSharedTensor"]): self@y

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: Raised when private matmul is performed parties!=3.

        """
        y_tensor, session = self.sanity_checks(self, y)
        is_private = isinstance(y, ReplicatedSharedTensor)

        op_str = "matmul"

        if is_private:
            if session.nr_parties == 3:
                from sympc.protocol import Falcon

                result = [Falcon.multiplication_protocol(self, y_tensor, op_str)]
            else:
                raise ValueError(
                    "Private matmul between ReplicatedSharedTensors is allowed only for 3 parties"
                )
        else:
            result = [
                operator.matmul(share, y_tensor.shares[0]) for share in self.shares
            ]

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

        res = ReplicatedSharedTensor(
            session_uuid=self.session_uuid, config=self.config, ring_size=self.ring_size
        )
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
            ValueError : If invalid shift value is provided.
        """
        if not isinstance(y, int):
            raise ValueError("Right Shift works only with integers!")

        ring_bits = get_nr_bits(self.ring_size)
        if y < 0 or y > ring_bits - 1:
            raise ValueError(
                f"Invalid value for right shift: {y}, must be in range:[0,{ring_bits-1}]"
            )

        res = ReplicatedSharedTensor(
            session_uuid=self.session_uuid, config=self.config, ring_size=self.ring_size
        )
        res.shares = [share >> y for share in self.shares]
        return res

    def bit_extraction(self, pos: int = 0) -> "ReplicatedSharedTensor":
        """Extracts the bit at the specified position.

        Args:
            pos (int): position to extract bit.

        Returns:
            ReplicatedSharedTensor : extracted bits at specific position.

        Raises:
            ValueError: If invalid position is provided.
        """
        ring_bits = get_nr_bits(self.ring_size)
        if pos < 0 or pos > ring_bits - 1:
            raise ValueError(
                f"Invalid position for bit_extraction: {pos}, must be in range:[0,{ring_bits-1}]"
            )
        shares = []
        # logical shift
        bit_mask = torch.ones(self.shares[0].shape, dtype=self.shares[0].dtype) << pos
        shares = [share & bit_mask for share in self.shares]
        rst = ReplicatedSharedTensor(
            shares=shares,
            session_uuid=self.session_uuid,
            config=Config(encoder_base=1, encoder_precision=0),
            ring_size=2,
        )
        return rst

    def rmatmul(self, y):
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y: self@y

        Raises:
            NotImplementedError: Raised when implementation not present
        """
        raise NotImplementedError

    def xor(
        self, y: Union[int, torch.Tensor, "ReplicatedSharedTensor"]
    ) -> "ReplicatedSharedTensor":
        """Apply the "xor" operation between "self" and "y".

        Args:
            y: public bit

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError : If ring size is invalid.
        """
        if self.ring_size == 2:
            return self + y
        elif self.ring_size in RING_SIZE_TO_TYPE:
            return self + y - (self * y * 2)
        else:
            raise ValueError(f"The ring_size {self.ring_size} is not supported.")

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

        if self.config != y.config:
            return False

        if self.session_uuid and y.session_uuid and self.session_uuid != y.session_uuid:
            return False

        if self.ring_size != y.ring_size:
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
    def shares_sum(shares: List[torch.Tensor], ring_size: int) -> torch.Tensor:
        """Returns sum of tensors based on ring_size.

        Args:
            shares (List[torch.Tensor]) : List of tensors.
            ring_size (int): Ring size of share associated with the tensors.

        Returns:
            value (torch.Tensor): sum of the tensors.
        """
        if ring_size == 2:
            return reduce(lambda x, y: x ^ y, shares)
        elif ring_size == PRIME_NUMBER:
            return reduce(ReplicatedSharedTensor.addmodprime, shares)
        else:
            return sum(shares)

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
        if not ispointer(share_ptr):
            return share_ptr
        elif not islocal(share_ptr):
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

        ring_size = local_shares[0].ring_size

        return ReplicatedSharedTensor.shares_sum(shares, ring_size)

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
        ring_size = local_shares[0].ring_size
        shares_sum = ReplicatedSharedTensor.shares_sum

        all_shares = [rst.shares for rst in local_shares]
        # reconstruct shares from all parties and verify
        value = None
        for party_rank in range(nparties):
            tensor = shares_sum(
                [all_shares[party_rank][0]] + all_shares[(party_rank + 1) % (nparties)],
                ring_size,
            )

            if value is None:
                value = tensor
            elif (tensor != value).any():
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
        ring_size: int,
        config: Config,
    ) -> "ReplicatedSharedTensor":
        """Distributes shares to party.

        Args:
            shares (List[Union[ShareTensor,torch.Tensor]]): Shares to be distributed.
            party_rank (int): Rank of party.
            session (Session): Current session
            ring_size (int): Ring size of tensor to distribute
            config (Config): The configuration(base,precision) of the tensor.

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
            session_uuid=session.rank_to_uuid[party_rank],
            config=config,
            ring_size=ring_size,
        )
        tensor.shares = party_shares
        return tensor.send(party)

    @staticmethod
    def distribute_shares(
        shares: List[Union[ShareTensor, torch.Tensor]],
        session: Session,
        ring_size: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> List["ReplicatedSharedTensor"]:
        """Distribute a list of shares.

        Args:
            shares (List[ShareTensor): list of shares to distribute.
            session (Session): Session.
            ring_size (int): ring_size the shares belong to.
            config (Config): The configuration(base,precision) of the tensor.

        Returns:
            List of ReplicatedSharedTensors.

        Raises:
            TypeError: when Datatype of shares is invalid.

        """
        if not isinstance(shares, (list, tuple)):
            raise TypeError("Shares to be distributed should be a list of shares")

        if len(shares) != session.nr_parties:
            return ValueError(
                "Number of shares to be distributed should be same as number of parties"
            )

        if ring_size is None:
            ring_size = session.ring_size
        if config is None:
            config = session.config

        args = [
            [shares, party_rank, session, ring_size, config]
            for party_rank in range(session.nr_parties)
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
                session_uuid=_self.session_uuid,
                config=_self.config,
                ring_size=_self.ring_size,
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
                session_uuid=_self.session_uuid,
                config=_self.config,
                ring_size=_self.ring_size,
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
