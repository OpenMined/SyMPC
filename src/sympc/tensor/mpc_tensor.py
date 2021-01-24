"""Class used to have orchestrate the computation on shared values."""

# stdlib
from functools import lru_cache
import operator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from syft.core.node.common.client import Client
import torch
import torchcsprng as csprng  # type: ignore

from sympc.encoder import FixedPointEncoder
from sympc.session import Session
from sympc.tensor import ShareTensor
from sympc.utils import islocal
from sympc.utils import ispointer
from sympc.utils import parallel_execution


class MPCTensor:
    """This class is used by an orchestrator that wants to do computation on
    data it does not see.

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


    Attributes:
        share_ptrs (List[ShareTensor]): pointer to the shares (hold by the parties)
        session (Session): session used for the MPC
        shape (Union[torch.size, tuple]): the shape for the shared secret
    """

    __slots__ = {"share_ptrs", "session", "shape"}

    def __init__(
        self,
        session: Optional[Session] = None,
        secret: Optional[Union[ShareTensor, torch.Tensor, float, int]] = None,
        shape: Optional[Union[torch.Size, List[int], Tuple[int, ...]]] = None,
        shares: Optional[List[ShareTensor]] = None,
    ) -> None:
        """Initializer for the MPCTensor (ShareTensorControlCenter It can be
        used in two ways:

        - secret is known by the orchestrator
        - secret is not known by the orchestrator (PRZS is employed)
        """

        if session is None and secret.session is None:
            raise ValueError(
                "Need to provide a session, as argument or in the ShareTensor"
            )

        self.session = session if session is not None else secret.session

        if len(self.session.session_ptrs) == 0:
            raise ValueError("setup_mpc was not called on the session")

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
                self.share_ptrs = MPCTensor.generate_przs(self.shape, self.session)
                for i, share in enumerate(self.share_ptrs):
                    if share.client == secret.client:  # type: ignore
                        self.share_ptrs[i] = self.share_ptrs[i] + secret
                        return
            else:
                tensor_type = self.session.tensor_type
                shares = MPCTensor.generate_shares(
                    secret, self.session.nr_parties, tensor_type
                )

        if not ispointer(shares[0]):
            shares = MPCTensor.distribute_shares(shares, self.session.parties)

        if shape is not None:
            self.shape = shape

        self.share_ptrs = shares

    @staticmethod
    def distribute_shares(shares: List[ShareTensor], parties: List[Client]):
        """Distribute a list of shares.

        Args:
            shares (List[ShareTensor): list of shares to distribute.
            parties (List[Client]): list to parties to distribute.

        Returns:
            List of ShareTensorPointers.
        """
        share_ptrs = []
        for share, party in zip(shares, parties):
            share_ptrs.append(share.send(party))

        return share_ptrs

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
        """Sanity check to validate that a new instance for MPCTensor can be
        created.

        Args:
            secret (Union[ShareTensor, torch.Tensor, float, int]): secret to check
            shape (Optional[Union[torch.Size, List[int], Tuple[int, ...]]]): shape of the secret.
                Mandatory if secret is at another party.
            session (Session): session

        Returns:
            Tuple representing the ShareTensor, the shape, boolean if the secret is remote or local.
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
                secret = ShareTensor(data=secret, session=session)

            shape = secret.shape

        return secret, shape, is_remote_secret

    @staticmethod
    def generate_przs(
        shape: Union[torch.Size, List[int], Tuple[int, ...]], session: Session
    ) -> List[ShareTensor]:
        """Generate Pseudo-Random-Zero Shares at the parties involved in the
        computation.

        Args:
            shape (Union[torch.Size, List[int], Tuple[int, ...]]): shape of the tensor.
            session: session.

        Returns:
            List[ShareTensor]. List of Pseudo-Random-Zero Shares
        """

        shape = tuple(shape)

        shares = []
        for session_ptr, generators_ptr in zip(
            session.session_ptrs, session.przs_generators
        ):
            share_ptr = session_ptr.przs_generate_random_share(shape, generators_ptr)
            shares.append(share_ptr)

        return shares

    @staticmethod
    def generate_shares(
        secret: Union[ShareTensor, torch.Tensor, float, int],
        nr_parties: int,
        tensor_type: Optional[torch.dtype] = None,
        **kwargs,
    ) -> List[ShareTensor]:
        """Given a secret, split it into a number of shares such that each
        party would get one.

        Args:
            secret (Union[ShareTensor, torch.Tensor, float, int]): secret to split
            nr_parties (int): number of parties to split the scret
            tensor_type (torch.dtype, optional): tensor type. Defaults to None.
            **kwargs: keywords arguments passed to ShareTensor

        Returns:
            List[ShareTensor]. List of ShareTensor

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

        if not isinstance(secret, (ShareTensor, torch.Tensor, float, int)):
            raise ValueError(
                "Secret should be a ShareTensor, torchTensor, float or int."
            )

        # if secret is not a ShareTensor, a new instance is created
        if isinstance(secret, (torch.Tensor, float, int)):
            secret = ShareTensor(secret, **kwargs)

        shape = secret.shape

        random_shares = []
        generator = csprng.create_random_device_generator()

        for _ in range(nr_parties - 1):
            rand_value = torch.empty(size=shape, dtype=tensor_type).random_(
                generator=generator
            )
            share = ShareTensor(session=secret.session)
            share.tensor = rand_value

            random_shares.append(share)

        shares = []
        for i in range(nr_parties):
            if i == 0:
                share = random_shares[i]
            elif i < nr_parties - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]

            shares.append(share)
        return shares

    def reconstruct(
        self, decode: bool = True, get_shares: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Request and get the shares from all the parties and reconstruct the
        secret. Depending on the value of "decode", the secret would be decoded
        or not using the FixedPrecision Encoder specific for the session.

        Args:
            decode (bool): True if decode using FixedPointEncoder. Defaults to True
            get_shares (boot): True if get shares. Defaults to False.

        Returns:
            torch.Tensor. The secret reconstructed.
        """

        def _request_and_get(share_ptr: ShareTensor) -> ShareTensor:
            """Function used to request and get a share - Duet Setup

            Args:
                share_ptr (ShareTensor): a ShareTensor

            Returns:
                ShareTensor. The ShareTensor in local.

            """

            if not islocal(share_ptr):
                share_ptr.request(name="reconstruct", block=True)
            res = share_ptr.get_copy()
            return res

        request = _request_and_get

        request_wrap = parallel_execution(request)

        args = [[share] for share in self.share_ptrs]
        local_shares = request_wrap(args)

        tensor_type = self.session.tensor_type

        shares = [share.tensor for share in local_shares]

        if get_shares:
            return shares

        plaintext = sum(shares)

        if decode:
            fp_encoder = FixedPointEncoder(
                base=self.session.config.encoder_base,
                precision=self.session.config.encoder_precision,
            )

            plaintext = fp_encoder.decode(plaintext)

        return plaintext

    def get_shares(self):
        """Get the shares."""
        res = self.reconstruct(get_shares=True)
        return res

    def add(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self + y

        Returns:
            MPCTensor. Result of the operation.
        """

        return self.__apply_op(y, "add")

    def sub(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "sub" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self - y

        Returns:
            MPCTensor. Result of the operation.
        """

        return self.__apply_op(y, "sub")

    def rsub(self, y: Union[torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "sub" operation between "y" and "self".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self - y

        Returns:
            MPCTensor. Result of the operation.
        """

        return self.__apply_op(y, "sub") * -1

    def mul(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self * y

        Returns:
            MPCTensor. Result of the operation.
        """

        return self.__apply_op(y, "mul")

    def matmul(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self @ y

        Returns:
            MPCTensor. Result of the operation.
        """

        return self.__apply_op(y, "matmul")

    def rmatmul(self, y: torch.Tensor) -> "MPCTensor":
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: y @ self

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
        result.shape = MPCTensor.__get_shape("matmul", y_shape, self.shape)

        scale = (
            self.session.config.encoder_base ** self.session.config.encoder_precision
        )
        result = result.div(scale)

        return result

    def div(self, y: Union["MPCTensor", torch.Tensor, float, int]) -> "MPCTensor":
        """Apply the "div" operation between "self" and "y".

        Args:
            y (Union["MPCTensor", torch.Tensor, float, int]: self / y

        Returns:
            MPCTensor. Result of the operation.
        """

        is_private = isinstance(y, MPCTensor)
        if is_private:
            raise NotImplementedError("Not implemented for MPCTensor")

        from sympc.protocol.spdz import spdz

        result = spdz.public_divide(self, y)
        return result

    def __apply_private_op(self, y: "MPCTensor", op_str: str) -> "MPCTensor":
        """Apply an operation on 2 MPCTensor (secret shared values)

        Args:
            y (MPCTensor): tensor to apply the operation
            op_str (str): the operation

        Returns:
            MPCTensor. The operation "op_str" applied on "self" and "y"
        """

        if y.session.uuid != self.session.uuid:
            raise ValueError(
                f"Need same session {self.session.uuid} and {y.session.uuid}"
            )

        if op_str in {"mul", "matmul"}:
            from sympc.protocol.spdz import spdz

            result = spdz.mul_master(self, y, op_str)
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            shares = [
                op(*share_tuple) for share_tuple in zip(self.share_ptrs, y.share_ptrs)
            ]

            result = MPCTensor(shares=shares, session=self.session)

        return result

    def __apply_public_op(
        self, y: Union[torch.Tensor, float, int], op_str: str
    ) -> "MPCTensor":
        """Apply an operation on "self" which is a MPCTensor and a public
        value.

        Args:
            y (Union[torch.Tensor, float, int]): tensor to apply the operation.
            op_str (str): the operation.

        Returns:
            MPCTensor. The operation "op_str" applied on "self" and "y"
        """

        op = getattr(operator, op_str)
        if op_str in {"mul", "matmul"}:
            shares = [op(share, y) for share in self.share_ptrs]
        elif op_str in {"add", "sub"}:
            shares = list(self.share_ptrs)
            # Only the rank 0 party has to add the element
            shares[0] = op(shares[0], y)
        else:
            raise ValueError(f"{op_str} not supported")

        result = MPCTensor(shares=shares, session=self.session)
        return result

    @staticmethod
    @lru_cache(maxsize=128)
    def __get_shape(
        op_str: str, x_shape: Tuple[int], y_shape: Tuple[int]
    ) -> Tuple[int]:

        if x_shape is None or y_shape is None:
            raise ValueError(
                f"Shapes should not be None; x_shape {x_shape}, y_shape {y_shape}"
            )

        op = getattr(operator, op_str)
        x = torch.ones(size=x_shape)
        y = torch.ones(size=y_shape)

        res = op(x, y)
        return res.shape

    def __apply_op(
        self, y: Union["MPCTensor", torch.Tensor, float, int], op_str: str
    ) -> "MPCTensor":
        """Apply an operation on "self" which is a MPCTensor "y" This function
        checks if "y" is private or public value.

        Args:
            y: tensor to apply the operation.
            op_str: the operation.

        Returns:
            MPCTensor. the operation "op_str" applied on "self" and "y"
        """

        is_private = isinstance(y, MPCTensor)

        if is_private:
            result = self.__apply_private_op(y, op_str)
        else:
            result = self.__apply_public_op(y, op_str)

        if isinstance(y, (float, int)):
            y_shape = (1,)
        else:
            y_shape = y.shape

        result.shape = MPCTensor.__get_shape(op_str, self.shape, y_shape)

        if op_str in {"mul", "matmul"} and not (
            is_private and self.session.nr_parties == 2
        ):
            # For private op we do the division in the mul_parties function from spdz
            scale = (
                self.session.config.encoder_base
                ** self.session.config.encoder_precision
            )
            result = result.div(scale)

        return result

    def __str__(self) -> str:
        """Return the string representation of MPCTensor."""
        type_name = type(self).__name__
        out = f"[{type_name}]\nShape: {self.shape}"

        for share in self.share_ptrs:
            out = f"{out}\n\t| {share.client} -> {share.__name__}"
        return out

    def __repr__(self):
        return self.__str__()

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = div
