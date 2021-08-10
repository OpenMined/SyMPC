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
import numpy as np
import torch

import sympc
from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.tensor import ShareTensor
from sympc.utils import RING_SIZE_TO_TYPE
from sympc.utils import get_type_from_ring
from sympc.utils import islocal
from sympc.utils import parallel_execution

from .tensor import SyMPCTensor

PROPERTIES_NEW_RS_TENSOR: Set[str] = {"T"}
METHODS_NEW_RS_TENSOR: Set[str] = {"unsqueeze", "view", "t", "sum", "clone", "repeat","flatten","expand"}
BINARY_MAP = {"add": "xor", "sub": "xor", "mul": "and_"}
SIGNED_MAP = {
    "bool": "bool",
    "uint8": "int8",
    "uint16": "int16",
    "uint32": "int32",
    "uint64": "int64",
}

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
    METHODS_FORWARD = {"numel", "t", "unsqueeze", "view", "sum", "clone", "repeat","flatten","expand"}
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
          if(type(shares[0])!=list):
             self.shares = [self._encode(share).to(tensor_type) for share in shares]
          else:
             self.shares = [share for share in shares]

        """if shares is not None:
            for i in range(len(shares)):
                share = shares[i]
                if isinstance(shares[i], torch.Tensor):
                    share = self._encode(share).to(tensor_type)
                    self.shares.append(share)
                else:
                    self.shares.append(share)"""
                
                
    def reshape(self, lst, shape):
        # stdlib
        from functools import reduce
        from operator import mul

        if len(shape) == 1:
            return lst
        n = reduce(mul, shape[1:])
        return [
            self.reshape(lst[i * n : (i + 1) * n], shape[1:])
            for i in range(len(lst) // n)
        ]
    
    
    def share_matrix(self):
        tensors = []
        for i in range(0,self.shares[0].shape[0]):
           subtensors=[]
           for j in range(0,self.shares[0].shape[1]):

                share = [self.shares[0][i][j], self.shares[1][i][j]]
                tensor = ReplicatedSharedTensor(
                    shares=share,
                    session_uuid=self.session_uuid,
                    ring_size=self.ring_size,config=Config(encoder_base=1, encoder_precision=0)
                )
                subtensors.append(tensor)
                
           tensors.append(subtensors)
                
        return ReplicatedSharedTensor(shares=tensors,session_uuid=self.session_uuid,
                    ring_size=self.ring_size,config=Config(encoder_base=1, encoder_precision=0)
                )
        

    """def share_matrix(self):
        
        shape=self.shape
        
        shares=self.shares
        
        if(len(shares[0].shape)==4):
            
            timesteps,batches,w,h=shape
            shares[0]=shares[0].reshape((timesteps*batches,w,h))
            shares[1]=shares[1].reshape((timesteps*batches,w,h))
        
        if(len(shares[0].shape)==3):
            
            batches,w,h=shape
            shares[0]=shares[0].reshape((batches,w,h))
            shares[1]=shares[1].reshape((batches,w,h))
        
        elif(len(shares[0].shape)==2):
            
            w,h=shape
            shares[0]=shares[1].reshape((1,w,h))
            shares[1]=shares[1].reshape((1,w,h))
    
        batches=[]
        
    
        for batch in range(0, shares[0].shape[0]):
          tensors = []
          for i in range(0, shares[0].shape[1]):
            for j in range(0, shares[0].shape[2]):

                share = [shares[0][batch][i][j], shares[1][batch][i][j]]
                tensor = ReplicatedSharedTensor(
                    shares=share,
                    config=Config(encoder_base=1, encoder_precision=0),
                    session_uuid=self.session_uuid,
                    ring_size=self.ring_size,
                )
                tensors.append(tensor)

          tensors = self.reshape(tensors,(w,h))
          
          
          batches.append(ReplicatedSharedTensor(
            shares=tensors,
            config=Config(encoder_base=1, encoder_precision=0),
            session_uuid=self.session_uuid,
            ring_size=self.ring_size,
          ))
         
          
          
        #Reshape for time series
        return batches"""

    def matrix_to_rst(self):
                
        tensors=[]
        
        shape=(len(self.shares),len(self.shares[0]))
        
        shares=self.shares
        
        tensor1 = torch.zeros([shape[0],shape[1]])
        tensor2 = torch.zeros([shape[0],shape[1]])

        for i in range(0, shape[0]):
            for j in range(0,shape[1]):
                tensor1[i][j] = self.shares[i][j].shares[0]
                tensor2[i][j] = self.shares[i][j].shares[1]
                
        shares=[tensor1,tensor2]    

        return ReplicatedSharedTensor(
            shares=shares,
            session_uuid=self.session_uuid,
            ring_size=self.ring_size,config=Config(encoder_base=1, encoder_precision=0)
        )
    
    def extend(self,data):
    
        print(self.shares[0].shape)    
    
        in_shape=self.shares[0].shape
        data_shape=data.shares[0].shape
        
        in_share1=self.shares[0]
        in_share2=self.shares[1]
        
        data_share1=data.shares[0]
        data_share2=data.shares[1]
        
        if(len(in_shape)==0):
          in_share1=in_share1.reshape([1])
          in_share2=in_share2.reshape([1])
          
        if(len(data_shape)==0):
          data_share1=data_share1.reshape([1])
          data_share2=data_share2.reshape([1])
         
        a=torch.cat([in_share1,in_share2])
        b=torch.cat([data_share1,data_share2])
          
        return ReplicatedSharedTensor(
            shares=[a,b],
            session_uuid=self.session_uuid,
            ring_size=self.ring_size,config=Config(encoder_base=1, encoder_precision=0)
          )

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
    def addmodprime(
        x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Computes addition(x+y) modulo PRIME_NUMBER constant.

        Args:
            x (Union[torch.Tensor,np.ndarray]): input tensor
            y (Union[torch.Tensor,np.ndarray]): input tensor

        Returns:
            value (Union[torch.Tensor,np.ndarray]): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not uint8 type.
            ValueError: If input tensor does not match a tensor type.
        """
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.uint8 or y.dtype != torch.uint8:
                raise ValueError(
                    f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
                )
        elif isinstance(x, np.ndarray):
            if x.dtype != np.uint8 or y.dtype != np.uint8:
                raise ValueError(
                    f"Both numpy tensors x:{x.dtype} y:{y.dtype} should be of np.uint8 dtype"
                )
        else:
            raise ValueError("Invalid input")

        return (x + y) % PRIME_NUMBER

    @staticmethod
    def submodprime(
        x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Computes subtraction(x-y) modulo PRIME_NUMBER constant.

        Args:
            x (Union[torch.Tensor,np.ndarray]): input tensor
            y (Union[torch.Tensor,np.ndarray]): input tensor

        Returns:
            value (Union[torch.Tensor,np.ndarray]): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not uint8 type.
            ValueError: If input tensor does not match a tensor type.
        """
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.uint8 or y.dtype != torch.uint8:
                raise ValueError(
                    f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
                )

            # Typecasting is done, as underflow returns a positive number,as it is unsigned.
            x = x.to(torch.int8)
            y = y.to(torch.int8)

            result = (x - y) % PRIME_NUMBER
            result = result.to(torch.uint8)

        elif isinstance(x, np.ndarray):
            if x.dtype != np.uint8 or y.dtype != np.uint8:
                raise ValueError(
                    f"Both numpy tensors x:{x.dtype} y:{y.dtype} should be of np.uint8 dtype"
                )
            # Typecasting is done, as underflow returns a positive number,as it is unsigned.
            x = x.astype(np.int8)
            y = y.astype(np.int8)

            result = (x - y) % PRIME_NUMBER
            result = result.astype(np.uint8)

        else:
            raise ValueError("Invalid input")

        return result

    @staticmethod
    def mulmodprime(
        x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Computes multiplication(x*y) modulo PRIME_NUMBER constant.

        Args:
            x (Union[torch.Tensor,np.ndarray]): input tensor
            y (Union[torch.Tensor,np.ndarray]): input tensor

        Returns:
            value (Union[torch.Tensor,np.ndarray]): Result of the operation.

        Raises:
            ValueError : If either of the tensors datatype is not uint8 type.
            ValueError: If input tensor does not match a tensor type.
        """
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.uint8 or y.dtype != torch.uint8:
                raise ValueError(
                    f"Both tensors x:{x.dtype} y:{y.dtype} should be of torch.uint8 dtype"
                )
            # We typecast as multiplication result in 2n bits ,which causes overflow.
            x = x.to(torch.int16)
            y = y.to(torch.int16)

            result = (x * y) % PRIME_NUMBER
            result = result.to(torch.uint8)

        elif isinstance(x, np.ndarray):
            if x.dtype != np.uint8 or y.dtype != np.uint8:
                raise ValueError(
                    f"Both numpy tensors x:{x.dtype} y:{y.dtype} should be of np.uint8 dtype"
                )
            # We typecast as multiplication result in 2n bits ,which causes overflow.
            x = x.astype(np.int16)
            y = y.astype(np.int16)

            result = (x * y) % PRIME_NUMBER
            result = result.astype(np.uint8)

        else:
            raise ValueError("Invalid input")

        return result

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

            if isinstance(y, np.ndarray):
                # use the encoding of torch tensor
                # TODO:should have separate encoding for numpy.
                y = torch.from_numpy(y.astype(SIGNED_MAP[str(y.dtype)]))

            y = ReplicatedSharedTensor(
                session_uuid=x.session_uuid,
                shares=[y],
                ring_size=x.ring_size,
                config=x.config,
            )

            if isinstance(x.shares[0], np.ndarray) and isinstance(
                y.shares[0], torch.Tensor
            ):
                y.shares[0] = y.shares[0].numpy().astype(x.shares[0].dtype)

        elif y.session_uuid and x.session_uuid and y.session_uuid != x.session_uuid:
            raise ValueError(
                f"Session UUIDs did not match {x.session_uuid} {y.session_uuid}"
            )
        elif len(x.shares) != len(y.shares):
            raise ValueError("Both RSTensors should have equal number of shares.")
        elif x.ring_size != y.ring_size:
            raise ValueError("Both RSTensors should have same ring_size")

        if isinstance(x.shares[0], np.ndarray) and not isinstance(
            y.shares[0], np.ndarray
        ):
            y = y.to_numpy(str(x.shares[0].dtype))

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
        """
        if not isinstance(y, int):
            raise ValueError("Right Shift works only with integers!")

        res = ReplicatedSharedTensor(
            session_uuid=self.session_uuid, config=self.config, ring_size=self.ring_size
        )
        res.shares = [share >> y for share in self.shares]
        return res

    def lshift(self, y: int) -> "ReplicatedSharedTensor":
        """Apply the "lshift" operation to "self".

        Args:
            y (int): shift value

        Returns:
            ReplicatedSharedTensor: Result of the operation.

        Raises:
            ValueError: If y is not an integer.
        """
        if not isinstance(y, int):
            raise ValueError("Left Shift works only with integers!")

        res = ReplicatedSharedTensor(
            session_uuid=self.session_uuid, config=self.config, ring_size=self.ring_size
        )
        res.shares = [share << y for share in self.shares]
        return res

    def bit_extraction(self, pos: int = 0) -> "ReplicatedSharedTensor":
        """Extracts the bit at the specified position.

        Args:
            pos (int): position to extract bit.

        Returns:
            ReplicatedSharedTensor : extracted bits at specific position.
        """
        shares = []
        # logical shift
        bit_mask = torch.ones(self.shares[0].shape, dtype=self.shares[0].dtype) << pos
        for share in self.shares:
            tensor = share & bit_mask
            shares.append(tensor)
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

            if isinstance(share, torch.Tensor) or isinstance(share, np.ndarray):
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
        if not isinstance(shares, list):
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

    def to_numpy(self, dtype: str) -> "ReplicatedSharedTensor":
        """Converts the underlying tensor shares to numpy array.

        Args:
            dtype (str) : The data type to convert the tensor to.

        Returns:
            ReplicatedSharedTensor: converted RSTensor object.
        """
        rst = self.clone()
        for idx, share in enumerate(self.shares):
            rst.shares[idx] = share.numpy().astype(dtype)

        return rst

    def from_numpy(self) -> "ReplicatedSharedTensor":
        """Converts the underlying tensor to torch tensor.

        Returns:
            ReplicatedSharedTensor: converted RSTensor object.
        """
        shares = copy.deepcopy(self.shares)
        dtype = str(self.shares[0].dtype)
        # Convert unsigned to signed as torch supports only signed.
        dtype = SIGNED_MAP.get(dtype, dtype)
        rst = ReplicatedSharedTensor(
            session_uuid=self.session_uuid, config=self.config, ring_size=self.ring_size
        )
        rst.shares = []
        for idx, share in enumerate(shares):
            rst.shares.append(torch.from_numpy(share.astype(dtype)))

        return rst

    def wrap_rst(self, other) -> "ReplicatedSharedTensor":
        """Applies wrap2 on the shares.

        Args:
            other (ReplicatedSharedTensor): tensor to wrap on.

        Returns:
            result (ReplicatedSharedTensor): Wrap of the tensor in binary.

        Raises:
            ValueError: If the input tensors are not numpy array.
        """
        from sympc.protocol import Falcon

        if not (
            isinstance(self.shares[0], np.ndarray)
            and isinstance(other.shares[0], np.ndarray)
        ):
            raise ValueError("Input tensor to wrap should be a numpy array.")
        shares = []

        for x_share, y_share in zip(self.shares, other.shares):
            shares.append(Falcon.wrap2(x_share, y_share))

        config = Config(encoder_base=1, encoder_precision=0)
        result = ReplicatedSharedTensor(
            ring_size=2,
            session_uuid=self.session_uuid,
            config=config,
        )
        result.shares = shares
        return result

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
    __lshift__ = lshift
