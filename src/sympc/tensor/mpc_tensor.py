"""Class used to orchestrate the computation on shared values."""

# stdlib
import functools
from functools import lru_cache
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import torch
import torchcsprng as csprng  # type: ignore

from sympc.approximations import APPROXIMATIONS
from sympc.config import Config
from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.tensor import ShareTensor
from sympc.utils import ispointer

from .tensor import SyMPCTensor

PROPERTIES_FORWARD_ALL_SHARES = {"T"}
METHODS_FORWARD_ALL_SHARES = {
    "t",
    "squeeze",
    "unsqueeze",
    "view",
    "sum",
    "clone",
    "flatten",
    "reshape",
    "repeat",
    "narrow",
    "dim",
    "transpose",
}
TRUNCATED_OPS = {"mul", "matmul", "conv2d", "conv_transpose2d"}


def wrapper_getattribute(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper to make sure that we call __getattribute__ before anything else.

    Args:
        func (Callable[Any, Any]): The function to wrap

    Returns:
        The wrapper for the func
    """

    def wrapper_func(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        """Use the getattr functionality to get the attribute/method.

        Args:
            *args (List[Any]): The attributes for the function call.
            **kwargs (Dict[str, Any]): The named attributes for the function call.

        Returns:
            The result of applying the function with args and kwargs.
        """
        _self, *new_args = args
        f = getattr(_self, func.__name__)
        res = f(*new_args, **kwargs)
        return res

    return wrapper_func


class MPCTensor(metaclass=SyMPCTensor):
    """Used by the orchestrator to compute on the shares.

    Arguments:
        session (Session): the session
        secret (Optional[Union[torch.Tensor, float, int]): in case the secret is
            known by the orchestrator it is split in shares and given to multiple
            parties
        shape (Optional[Union[torch.Size, tuple]): the shape of the secret in case
            the secret is not known by the orchestrator
            this is needed when a multiplication is needed between two secret values
            (need the shapes to be able to generate random elements in the proper way)
        shares (Optional[List[ShareTensor]]): in case the shares are already at the
             parties involved in the computation

    This class is used by an orchestrator that wants to do computation on
    data it does not see.

    Attributes:
        share_ptrs (List[ShareTensor]): pointer to the shares (hold by the parties)
        session (Session): session used for the MPC
        shape (Union[torch.size, tuple]): the shape for the shared secret
    """

    __slots__ = {
        "share_ptrs",
        "session",
        "shape",
        "grad",
        # We need this because only floating type tensor can have requires_grad
        # If not, we could use the self.tensor requires_grad
        # Use for training
        "requires_grad",
        "grad_fn",
        "ctx",
        "parents",
        "nr_out_edges",
    }

    # Used by the SyMPCTensor metaclass
    METHODS_FORWARD = {
        "numel",
        "t",
        "squeeze",
        "unsqueeze",
        "view",
        "sum",
        "clone",
        "flatten",
        "reshape",
        "repeat",
        "narrow",
        "dim",
        "transpose",
    }
    PROPERTIES_FORWARD = {"T"}

    def __init__(
        self,
        session: Session,
        secret: Optional[Union[ShareTensor, torch.Tensor, float, int]] = None,
        shape: Optional[Union[torch.Size, List[int], Tuple[int, ...]]] = None,
        shares: Optional[List[ShareTensor]] = None,
        requires_grad: bool = False,
    ) -> None:
        """Initializer for the MPCTensor. It can be used in two ways.

        ShareTensorControlCenter can be used in two ways:
        - secret is known by the orchestrator.
        - secret is not known by the orchestrator (PRZS is employed).

        Args:
            session (Session): The session used for the mpc computation
            secret (Optional[Union[ShareTensor, torch.Tensor, float, int]]): In case the secret is
                known by the orchestrator it is split in shares and given to multiple
                parties. Defaults to None.
            shape (Optional[Union[torch.Size, List[int], Tuple[int, ...]]]): The shape of the
                secret in case the secret is not known by the orchestrator this is needed
                when a multiplication is needed between two secret values (need the shapes
                to be able to generate random elements in the proper way). Defaults to None
            shares (Optional[List[ShareTensor]]): In case the shares are already at the
                parties involved in the computation. Defaults to None
            requires_grad: (bool): Specify if the MPCTensor is required for gradient computation

        Raises:
            ValueError: If session is not provided as argument or in the ShareTensor.
        """
        self.session = session

        if len(self.session.session_ptrs) == 0:
            raise ValueError("setup_mpc was not called on the session")

        self.shape = None

        if secret is not None:
            """In the case the secret is hold by a remote party then we use the
            PRZS to generate the shares and then the pointer tensor is added to
            share specific to the holder of the secret."""
            secret, self.shape, is_remote_secret = MPCTensor.sanity_checks(
                secret, shape, self.session
            )

            if is_remote_secret:
                # If the secret is remote we use PRZS (Pseudo-Random-Zero Shares) and the
                # party that holds the secret will add it to its share
                shares = MPCTensor.generate_przs(shape=self.shape, session=self.session)
                for i, share in enumerate(shares):
                    if share.client == secret.client:  # type: ignore
                        shares[i] = shares[i] + secret
                        break
            else:
                tensor_type = self.session.tensor_type
                shares = MPCTensor.generate_shares(
                    secret=secret,
                    config=self.session.config,
                    nr_parties=self.session.nr_parties,
                    tensor_type=tensor_type,
                )

        if not ispointer(shares[0]):
            shares = self.session.protocol.distribute_shares(shares, self.session)

        self.share_ptrs = shares

        if shape is not None:
            self.shape = shape

        # For training
        self.requires_grad = requires_grad
        self.nr_out_edges = 0
        self.ctx = {}

        self.grad = None
        self.grad_fn = None
        self.parents: List["MPCTensor"] = []

    @staticmethod
    def sanity_checks(
        secret: Union[ShareTensor, torch.Tensor, float, int],
        shape: Optional[Union[torch.Size, List[int], Tuple[int, ...]]],
        session: Session,
    ) -> Tuple[
        Union[ShareTensor, torch.Tensor, float, int],
        Union[torch.Size, List[int], Tuple[int, ...]],
        bool,
    ]:
        """Sanity check to validate that a new instance for MPCTensor can be created.

        Args:
            secret (Union[ShareTensor, torch.Tensor, float, int]): Secret to check.
            shape (Optional[Union[torch.Size, List[int], Tuple[int, ...]]]): shape of the secret.
                Mandatory if secret is at another party.
            session (Session): Session.

        Returns:
            Tuple representing the ShareTensor, the shape, boolean if the secret is remote or local.

        Raises:
            ValueError: If secret is at another party and shape is not specified.
        """
        is_remote_secret: bool = False

        if ispointer(secret):
            is_remote_secret = True
            if shape is None:
                raise ValueError(
                    "Shape must be specified if secret is at another party"
                )

            shape = shape
        else:
            if isinstance(secret, (int, float)):
                secret = torch.tensor(data=[secret])

            if isinstance(secret, torch.Tensor):
                secret = ShareTensor(data=secret, config=session.config)

            shape = secret.shape

        return secret, shape, is_remote_secret

    @staticmethod
    def generate_przs(
        shape: Union[torch.Size, List[int], Tuple[int, ...]],
        session: Session,
    ) -> List[ShareTensor]:
        """Generate Pseudo-Random-Zero Shares.

        PRZS at the parties involved in the computation.

        Args:
            shape (Union[torch.Size, List[int], Tuple[int, ...]]): Shape of the tensor.
            session (Session): Session.

        Returns:
            List[ShareTensor]. List of Pseudo-Random-Zero Shares.
        """
        shape = tuple(shape)

        shares = []
        for session_ptr in session.session_ptrs:
            share_ptr = session_ptr.przs_generate_random_share(shape=shape)
            shares.append(share_ptr)

        return shares

    @staticmethod
    def generate_shares(
        secret: Union[ShareTensor, torch.Tensor, float, int],
        nr_parties: int,
        config: Config = Config(),
        tensor_type: Optional[torch.dtype] = None,
    ) -> List[ShareTensor]:
        """Generate shares from secret.

        Given a secret, split it into a number of shares such that each
        party would get one.

        Args:
            secret (Union[ShareTensor, torch.Tensor, float, int]): Secret to split.
            nr_parties (int): Number of parties to split the scret.
            config (Config): Configuration used for the Share Tensor (in case it is needed).
                Use default Config if nothing provided. The ShareTensor config would have priority.
            tensor_type (torch.dtype, optional): tensor type. Defaults to None.

        Returns:
            List[ShareTensor]. List of ShareTensor.

        Raises:
            ValueError: If secret is not a expected format.

        Examples:
            >>> from sympc.tensor.mpc_tensor import MPCTensor
            >>> MPCTensor.generate_shares(secret=2, nr_parties=2)
            [[ShareTensor]
                | [FixedPointEncoder]: precision: 16, base: 2
                | Data: tensor([15511500.]), [ShareTensor]
                | [FixedPointEncoder]: precision: 16, base: 2
                | Data: tensor([-15380428.])]
            >>> MPCTensor.generate_shares(secret=2, nr_parties=2,
                encoder_base=3, encoder_precision=4)
            [[ShareTensor]
                | [FixedPointEncoder]: precision: 4, base: 3
                | Data: tensor([14933283.]), [ShareTensor]
                | [FixedPointEncoder]: precision: 4, base: 3
                | Data: tensor([-14933121.])]
        """
        if isinstance(secret, (torch.Tensor, float, int)):
            # if secret is not a ShareTensor, a new instance is created
            secret = ShareTensor(secret, config=config)
        else:
            config = secret.config

        if not isinstance(secret, ShareTensor):
            raise ValueError(
                "Secret should be a ShareTensor, torchTensor, float or int."
            )

        op = operator.sub
        shape = secret.shape

        random_shares = []
        generator = csprng.create_random_device_generator()

        for _ in range(nr_parties - 1):
            rand_value = torch.empty(size=shape, dtype=tensor_type).random_(
                generator=generator
            )
            share = ShareTensor(data=rand_value, config=config)
            share.tensor = rand_value

            random_shares.append(share)

        shares = []
        for i in range(nr_parties):
            if i == 0:
                share = random_shares[i]
            elif i < nr_parties - 1:
                share = op(random_shares[i], random_shares[i - 1])
            else:
                share = op(secret, random_shares[i - 1])

            shares.append(share)
        return shares

    def reconstruct(
        self, decode: bool = True, get_shares: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct the secret.

        Request and get the shares from all the parties and reconstruct the
        secret. Depending on the value of "decode", the secret would be decoded
        or not using the FixedPrecision Encoder specific for the session.

        Args:
            decode (bool): True if decode using FixedPointEncoder. Defaults to True
            get_shares (bool): Retrieve only shares.

        Returns:
            torch.Tensor. The secret reconstructed.
        """
        result = self.session.protocol.share_class.reconstruct(
            self.share_ptrs,
            get_shares=get_shares,
            security_type=self.session.protocol.security_type,
        )

        if get_shares:

            return result

        if decode:
            fp_encoder = FixedPointEncoder(
                base=self.session.config.encoder_base,
                precision=self.session.config.encoder_precision,
            )

            result = fp_encoder.decode(result)

        return result

    get = reconstruct

    def get_shares(self):
        """Get the shares.

        Returns:
            List[MPCTensor]: List of shares.
        """
        res = self.reconstruct(get_shares=True)
        return res

    def add(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self + y

        Returns:
            MPCTensor. Result of the operation.
        """
        return self.__apply_op(y, "add")

    def isub(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "isub" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self - y

        Returns:
            MPCTensor. Result of the operation.
        """
        res = self.__apply_op(y, "sub")
        self.share_ptrs = res.share_ptrs
        return self

    def sub(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "sub" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self - y

        Returns:
            MPCTensor. Result of the operation.
        """
        return self.__apply_op(y, "sub")

    def rsub(self, y: Union[torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "sub" operation between "y" and "self".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self - y

        Returns:
            MPCTensor. Result of the operation.
        """
        return self.__apply_op(y, "sub") * -1

    def mul(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self * y

        Returns:
            MPCTensor. Result of the operation.
        """
        return self.__apply_op(y, "mul")

    def matmul(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): self @ y

        Returns:
            MPCTensor. Result of the operation.
        """
        return self.__apply_op(y, "matmul")

    def conv2d(
        self,
        weight: Union["MPCTensor", torch.Tensor, float, int],
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> "MPCTensor":
        """Apply the "conv2d" operation between "self" and "y".

        Args:
            weight: the convolution kernel
            bias: optional bias
            stride: stride
            padding: padding
            dilation: dilation
            groups: groups

        Returns:
            MPCTensor. Result of the operation.
        """
        kwargs = {
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        bias = kwargs.pop("bias", None)

        convolution = self.__apply_op(weight, "conv2d", kwargs_=kwargs)

        if bias:
            return convolution + bias.unsqueeze(1).unsqueeze(1)
        else:
            return convolution

    def conv_transpose2d(
        self,
        weight: Union["MPCTensor", torch.Tensor, float, int],
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> "MPCTensor":
        """Apply the "conv_transpose2d" operation between "self" and "y".

        Args:
            weight: the convolution kernel
            bias: optional bias
            stride: stride
            padding: padding
            dilation: dilation
            groups: groups

        Returns:
            MPCTensor. Result of the operation.
        """
        kwargs = {
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        bias = kwargs.pop("bias", None)

        convolution = self.__apply_op(weight, "conv_transpose2d", kwargs_=kwargs)

        if bias:
            return convolution + bias.unsqueeze(1).unsqueeze(1)
        else:
            return convolution

    def rmatmul(self, y: torch.Tensor) -> "MPCTensor":
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): y @ self

        Returns:
            MPCTensor. Result of the operation.
        """
        op = getattr(operator, "matmul")
        shares = [op(y, share) for share in self.share_ptrs]

        if isinstance(y, (float, int)):
            y_shape = (1,)
        else:
            y_shape = y.shape

        result = MPCTensor(shares=shares, session=self.session)
        result.shape = MPCTensor._get_shape("matmul", y_shape, self.shape)

        scale = (
            self.session.config.encoder_base ** self.session.config.encoder_precision
        )
        result = result.truediv(scale)

        return result

    def rtruediv(self, y: Union[torch.Tensor, float, int]) -> "MPCTensor":
        """Apply recriprocal of MPCTensor.

        Args:
            y (Union[torch.Tensor, float, int]): Numerator.

        Returns:
            MPCTensor: Result of the operation.
        """
        reciprocal = APPROXIMATIONS["reciprocal"]
        return reciprocal(self) * y

    def truediv(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "div" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]): Denominator.

        Returns:
            MPCTensor: Result of the operation.

        Raises:
            ValueError: If parties are more than two.
        """
        is_private = isinstance(y, MPCTensor)

        # TODO: Implement support for more than two parties.
        if is_private:

            if len(self.session.session_ptrs) > 2:
                raise ValueError(
                    "Private division currently works with a maximum of two parties only."
                )

            reciprocal = APPROXIMATIONS["reciprocal"]
            return self.mul(reciprocal(y))

        from sympc.protocol.spdz import spdz

        result = spdz.public_divide(self, y)
        return result

    def pow(self, power: int) -> "MPCTensor":
        """Compute integer power of a number by recursion using mul.

        - Divide power by 2 and multiply base to itself (if the power is even)
        - Decrement power by 1 to make it even and then follow the first step

        Args:
            power (int): integer value to apply the operation

        Returns:
             MPCTensor: Result of the pow operation

        Raises:
            RuntimeError: if negative power is given
        """
        if power < 0:
            raise RuntimeError("Negative integer powers are not allowed.")

        base = self

        result = 1
        while power > 0:
            # If power is odd
            if power % 2 == 1:
                result = result * base

            # Divide the power by 2
            power = power // 2
            # Multiply base to itself
            base = base * base

        return result

    def __apply_private_op(
        self, y: "MPCTensor", op_str: str, kwargs_: Dict[Any, Any]
    ) -> "MPCTensor":
        """Apply an operation on 2 MPCTensor (secret shared values).

        Args:
            y (MPCTensor): Tensor to apply the operation
            op_str (str): The operation
            kwargs_ (dict): Kwargs for some operations like conv2d

        Returns:
            MPCTensor. The operation "op_str" applied on "self" and "y"

        Raises:
            ValueError: If session from MPCTensor and "y" is not the same.
            TypeError: If MPC tensors are not of same share class
            NotImplementedError: When op has not been implemented yet
        """
        if self.session.protocol.share_class != y.session.protocol.share_class:
            raise TypeError("Both MPC tensors should be of same share class.")

        if y.session.uuid != self.session.uuid:
            raise ValueError(
                f"Need same session {self.session.uuid} and {y.session.uuid}"
            )

        if op_str in TRUNCATED_OPS:
            from sympc.protocol import Falcon
            from sympc.protocol.spdz import spdz
            from sympc.tensor import ReplicatedSharedTensor

            if self.session.protocol.share_class == ShareTensor:
                result = spdz.mul_master(self, y, op_str, kwargs_)
                result.shape = MPCTensor._get_shape(op_str, self.shape, y.shape)

            elif self.session.protocol.share_class == ReplicatedSharedTensor:
                if op_str in {"mul", "matmul"}:
                    result = Falcon.mul_master(self, y, self.session, op_str, kwargs_)
                    result.shape = MPCTensor._get_shape(op_str, self.shape, y.shape)
                else:
                    raise NotImplementedError(
                        f"{op_str} has not implemented for ReplicatedSharedTensor"
                    )

            else:
                raise TypeError("Invalid Share Class")

        elif op_str == "xor":
            ring_size = int(self.share_ptrs[0].get_ring_size().get_copy())
            if ring_size == 2:
                return self + y
            else:
                return self + y - (self * y * 2)

        elif op_str in {"sub", "add"}:

            op = getattr(operator, op_str)
            shares = [
                op(*share_tuple) for share_tuple in zip(self.share_ptrs, y.share_ptrs)
            ]

            result = MPCTensor(shares=shares, shape=self.shape, session=self.session)

        return result

    def __apply_public_op(
        self, y: Union[torch.Tensor, float, int], op_str: str, kwargs_: Dict[Any, Any]
    ) -> "MPCTensor":
        """Apply an operation on "self" which is a MPCTensor and a public value.

        Args:
            y (Union[torch.Tensor, float, int]): Tensor to apply the operation.
            op_str (str): The operation.
            kwargs_ (dict): Kwargs for some operations like conv2d

        Returns:
            MPCTensor. The operation "op_str" applied on "self" and "y".

        Raises:
            ValueError: If "op_str" is not supported.
            TypeError: if share_class is not supported.
        """
        from sympc.tensor import ReplicatedSharedTensor

        op = getattr(operator, op_str)
        if op_str in {"mul", "matmul"}:
            shares = [op(share, y) for share in self.share_ptrs]

        elif op_str in {"add", "sub", "xor"}:
            shares = list(self.share_ptrs)
            # Only the rank 0 party has to add the element
            if self.session.protocol.share_class == ShareTensor:
                shares[0] = op(shares[0], y)
            elif self.session.protocol.share_class == ReplicatedSharedTensor:
                shares = [op(share, y) for share in self.share_ptrs]
            else:
                raise TypeError("Invalid Share Class")
        else:
            raise ValueError(f"{op_str} not supported")

        result = MPCTensor(shares=shares, session=self.session)
        return result

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_shape(
        op_str: str, x_shape: Tuple[int], y_shape: Tuple[int], **kwargs_: Dict[Any, Any]
    ) -> Tuple[int]:

        if x_shape is None or y_shape is None:
            raise ValueError(
                f"Shapes should not be None; x_shape {x_shape}, y_shape {y_shape}"
            )

        if op_str in ["conv2d", "conv_transpose2d"]:
            op = getattr(torch, op_str)
        else:
            op = getattr(operator, op_str)

        if op_str == "xor":
            x = torch.empty(size=x_shape, dtype=torch.bool)
            y = torch.empty(size=y_shape, dtype=torch.bool)
        else:
            x = torch.empty(size=x_shape)
            y = torch.empty(size=y_shape)

        res = op(x, y, **kwargs_)
        return res.shape

    def truncate(
        self, input_tensor: "MPCTensor", op_str: str, is_private: bool
    ) -> "MPCTensor":
        """Checks if operation requires truncation and performs it if required.

        Args:
            input_tensor (MPCTensor): Result of operation
            op_str (str): Operation name
            is_private (bool): If operation is private

        Returns:
            result (MPCTensor): Truncated result
        """
        from sympc.protocol import ABY3
        from sympc.tensor import ReplicatedSharedTensor

        result = None
        if (
            op_str in TRUNCATED_OPS
            and (not is_private or self.session.nr_parties > 2)
            and self.session.protocol.share_class == ShareTensor
        ):
            # For private op we do the division in the mul_parties function from spdz
            scale = (
                self.session.config.encoder_base
                ** self.session.config.encoder_precision
            )
            result = input_tensor.truediv(scale)
        elif (
            op_str in TRUNCATED_OPS
            and (not is_private)
            and self.session.protocol.share_class == ReplicatedSharedTensor
        ):
            ring_size = int(self.share_ptrs[0].get_ring_size().get_copy())
            conf_dict = self.share_ptrs[0].get_config().get_copy()
            config = Config(**conf_dict)

            result = ABY3.truncate(input_tensor, self.session, ring_size, config)

        else:
            result = input_tensor

        return result

    def __apply_op(
        self,
        y: Union["MPCTensor", torch.Tensor, float, int],
        op_str: str,
        kwargs_: Dict[Any, Any] = {},
    ) -> "MPCTensor":
        """Apply an operation on "self" which is a MPCTensor "y".

         This function checks if "y" is private or public value.

        Args:
            y: tensor to apply the operation.
            op_str: the operation.
            kwargs_ (dict): kwargs for some operations like conv2d

        Returns:
            MPCTensor. the operation "op_str" applied on "self" and "y"
        """
        is_private = isinstance(y, MPCTensor)

        if is_private:
            result = self.__apply_private_op(y, op_str, kwargs_)
        else:
            result = self.__apply_public_op(y, op_str, kwargs_)

        if isinstance(y, (float, int)):
            y_shape = (1,)
        else:
            y_shape = y.shape

        result.shape = MPCTensor._get_shape(op_str, self.shape, y_shape, **kwargs_)

        # Check operation and apply truncation if required.
        result = self.truncate(result, op_str, is_private)

        return result

    def __len__(self) -> int:
        """Return the length of MPCTensor.

        Returns:
            int: Length
        """
        return self.shape[0]

    def __str__(self) -> str:
        """Return the string representation of MPCTensor.

        Returns:
            str: String representation.
        """
        type_name = type(self).__name__
        out = f"[{type_name}]\nShape: {self.shape}"
        out = f"{out}\nRequires Grad: {self.requires_grad}"
        if self.grad_fn:
            out = f"{out}\nGradFunc: {self.grad_fn}"

        for share in self.share_ptrs:
            out = f"{out}\n\t| {share.client} -> {share.__name__}"

        return out

    def __repr__(self):
        """Representation.

        Returns:
            str: Representation.
        """
        return self.__str__()

    def __getattribute__(self, attr_name: str) -> Any:
        """Get the attribute and check if we should track the gradient.

        Args:
            attr_name (str): The name of the attribute

        Returns:
            The attribute specific for this instance
        """
        # TODO: Fix this
        from sympc.grads import GRAD_FUNCS
        from sympc.tensor.static import STATIC_FUNCS

        # Take the attribute and check if we need to assign a gradient function
        # Implementation similar to CrypTen
        grad_fn = GRAD_FUNCS.get(attr_name, None)
        session = object.__getattribute__(self, "session")
        if grad_fn and session.autograd_active:
            from sympc.grads import forward

            return functools.partial(forward, self, grad_fn)

        if attr_name in STATIC_FUNCS.keys():
            return functools.partial(STATIC_FUNCS[attr_name], self)

        approx_func = APPROXIMATIONS.get(attr_name, None)
        if approx_func is not None:
            return functools.partial(approx_func, self)

        return object.__getattribute__(self, attr_name)

    def backward(self, gradient: Optional["MPCTensor"] = None) -> None:
        """Perform the backward step on the computational graph.

        Args:
            gradient (MPCTensor): The gradient (received) from the computational graph

        Raises:
            ValueError: if there is no gradient function recorded for a node that needs to compute
                   the gradient
        """
        if not self.requires_grad:
            return

        self.session.autograd_active = False

        if gradient is None:
            if len(self.shape) not in {0, 1}:
                raise ValueError("Need to provide gradient if not scalar!")
            gradient = MPCTensor(secret=torch.ones(self.shape), session=self.session)

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        if len(self.parents) == 0:
            # We can not propagate from this node {self} because it does not have parents
            return

        self.nr_out_edges -= 1
        if self.nr_out_edges > 0:
            # For the moment we presume all parents are differentiable
            print("We will visit this node when all the parents returned the gradients")
            return

        if self.grad_fn is None:
            raise ValueError(f"Do not know how to propagate {self}")

        grad = self.grad_fn.backward(self.ctx, self.grad)
        if not isinstance(grad, (list, tuple)):
            grad = (grad,)

        for idx, parent in enumerate(self.parents):
            parent.backward(gradient=grad[idx])

        self.session.autograd_active = True

    @staticmethod
    def __check_or_convert(value, session) -> "MPCTensor":
        if not isinstance(value, MPCTensor):
            return MPCTensor(secret=value, session=session)
        else:
            return value

    @staticmethod
    def hook_property(property_name: str) -> Any:
        """Hook a framework property (only getter).

        Ex:
         * if we call "shape" we want to call it on the underlying share
        and return the result
         * if we call "T" we want to call it on all the underlying shares
        and wrap the result in an MPCTensor

        Args:
            property_name (str): property to hook

        Returns:
            A hooked property
        """

        def property_all_share_getter(_self: "MPCTensor") -> "MPCTensor":
            shares = []

            for share in _self.share_ptrs:
                prop = getattr(share, property_name)
                shares.append(prop)

            new_shape = getattr(torch.empty(_self.shape), property_name).shape
            res = MPCTensor(shares=shares, shape=new_shape, session=_self.session)
            return res

        def property_share_getter(_self: "MPCTensor") -> Any:
            prop = getattr(_self.share_ptrs[0], property_name)
            return prop

        if property_name in PROPERTIES_FORWARD_ALL_SHARES:
            res = property(property_all_share_getter, None)
        else:
            res = property(property_share_getter, None)

        return res

    @staticmethod
    def hook_method(method_name: str) -> Callable[..., Any]:
        """Hook a framework method.

        Ex:
         * if we call "numel" we want to forward it only to one share and return
        the result (without wrapping it in an MPCShare)
         * if we call "unsqueeze" we want to call it on all the underlying shares
        and we want to wrap those shares in a new MPCTensor

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_all_shares(
            _self: "MPCTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            shares = []

            for share in _self.share_ptrs:
                method = getattr(share, method_name)
                new_share = method(*args, **kwargs)
                shares.append(new_share)

                dummy_res = getattr(torch.empty(_self.shape), method_name)(
                    *args, **kwargs
                )
                new_shape = dummy_res.shape
            res = MPCTensor(shares=shares, shape=new_shape, session=_self.session)
            return res

        def method_share(
            _self: "MPCTensor", *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:
            method = getattr(_self.share_ptrs[0], method_name)
            res = method(*args, **kwargs)
            return res

        if method_name in METHODS_FORWARD_ALL_SHARES:
            res = method_all_shares
        else:
            res = method_share

        return res

    def le(self, other: "MPCTensor") -> "MPCTensor":
        """Lower or equal operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        protocol = self.session.get_protocol()
        other = self.__check_or_convert(other, self.session)
        return protocol.le(self, other)

    def ge(self, other: "MPCTensor") -> "MPCTensor":
        """Greater or equal operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        protocol = self.session.get_protocol()
        other = self.__check_or_convert(other, self.session)
        return protocol.le(other, self)

    def lt(self, other: "MPCTensor") -> "MPCTensor":
        """Lower than operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        protocol = self.session.get_protocol()
        other = self.__check_or_convert(other, self.session)
        fp_encoder = FixedPointEncoder(
            base=self.session.config.encoder_base,
            precision=self.session.config.encoder_precision,
        )
        one = fp_encoder.decode(1)
        return protocol.le(self + one, other)

    def gt(self, other: "MPCTensor") -> "MPCTensor":
        """Greater than operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        protocol = self.session.get_protocol()
        other = self.__check_or_convert(other, self.session)
        fp_encoder = FixedPointEncoder(
            base=self.session.config.encoder_base,
            precision=self.session.config.encoder_precision,
        )
        one = fp_encoder.decode(1)
        r = other + one
        return protocol.le(r, self)

    def eq(self, other: "MPCTensor") -> "MPCTensor":
        """Equal operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        protocol = self.session.get_protocol()
        other = self.__check_or_convert(other, self.session)
        return protocol.eq(self, other)

    def ne(self, other: "MPCTensor") -> "MPCTensor":
        """Not equal operator.

        Args:
            other (MPCTensor): MPCTensor to compare.

        Returns:
            MPCTensor: Result of the comparison.
        """
        other = self.__check_or_convert(other, self.session)
        return 1 - self.eq(other)

    def xor(self, y: Union["MPCTensor", torch.Tensor, int]) -> "MPCTensor":
        """XOR operator.

        Args:
            y (Union["MPCTensor", torch.Tensor, int]): MPCTensor to find xor.

        Returns:
            MPCTensor: Result of the xor.
        """
        return self.__apply_op(y, "xor")

    __add__ = wrapper_getattribute(add)
    __radd__ = wrapper_getattribute(add)
    __sub__ = wrapper_getattribute(sub)
    __isub__ = wrapper_getattribute(isub)
    __rsub__ = wrapper_getattribute(rsub)
    __mul__ = wrapper_getattribute(mul)
    __rmul__ = wrapper_getattribute(mul)
    __matmul__ = wrapper_getattribute(matmul)
    __rmatmul__ = wrapper_getattribute(rmatmul)
    __truediv__ = wrapper_getattribute(truediv)
    __rtruediv__ = wrapper_getattribute(rtruediv)
    __pow__ = wrapper_getattribute(pow)
    __le__ = wrapper_getattribute(le)
    __ge__ = wrapper_getattribute(ge)
    __lt__ = wrapper_getattribute(lt)
    __gt__ = wrapper_getattribute(gt)
    __eq__ = wrapper_getattribute(eq)
    __ne__ = wrapper_getattribute(ne)
    __xor__ = wrapper_getattribute(xor)


PARTIES_TO_SESSION: Dict[Any, Session] = {}


def share(_self, **kwargs: Dict[Any, Any]) -> MPCTensor:  # noqa
    session = None

    if "parties" not in kwargs and "session" not in kwargs:
        raise ValueError("Parties or Session should be provided as a kwarg")

    if "session" not in kwargs:
        parties = frozenset({client.id for client in kwargs["parties"]})

        if parties not in PARTIES_TO_SESSION:
            from sympc.session import SessionManager

            session = Session(kwargs["parties"])
            PARTIES_TO_SESSION[parties] = session
            SessionManager.setup_mpc(session)

            for key, val in kwargs.items():
                setattr(session, key, val)
        else:
            session = PARTIES_TO_SESSION[parties]

        kwargs.pop("parties")
        kwargs["session"] = session

    return MPCTensor(secret=_self, **kwargs)


METHODS_TO_ADD = [share]
