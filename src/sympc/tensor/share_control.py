"""
Class used to have orchestrate the computation on shared values
"""

import torch
import torchcsprng as csprng

from typing import Union
from typing import List
from typing import Tuple
from typing import Optional

from sympc.session import Session
from sympc.tensor import ShareTensor
from sympc.encoder import FixedPointEncoder
from sympc.utils import ispointer
from sympc.utils import isvm
from sympc.utils import parallel_execution

import operator


class ShareTensorCC:
    """
    This class is used by an orchestrator that wants to do computation
    on data it does not see.

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
        session (Sesssion): session used for the MPC
        shape (Union[torch.size, tuple]): the shape for the shared secret
    """

    __slots__ = {"share_ptrs", "session", "shape"}

    def __init__(
        self,
        session: Session,
        secret: Optional[Union[torch.Tensor, float, int]] = None,
        shape: Optional[Union[torch.Size, tuple]] = None,
        shares: Optional[List[ShareTensor]] = None,
    ) -> None:
        """Initializer for the ShareTensorCC (ShareTensorControlCenter
        It can be used in two ways:
        - secret is known by the orchestrator
        - secret is not known by the orchestrator (PRZS is employed)
        """

        if len(session.session_ptrs) == 0:
            raise ValueError("setup_mpc was not called on the session")

        self.session = session
        self.shape = None

        if secret is not None:
            secret, shape, is_remote_secret = ShareTensorCC.sanity_checks(
                secret, shape, session
            )
            parties = session.parties
            self.shape = shape

            if is_remote_secret:
                # If the secret is remote we use PRZS (Pseudo-Random-Zero Shares) and the
                # party that holds the secret will add it to it's share
                self.share_ptrs = ShareTensorCC.generate_przs(self.shape, self.session)
                for i, share in enumerate(self.share_ptrs):
                    if share.client == secret.client:
                        self.share_ptrs[i] = self.share_ptrs[i] + secret
                        break
            else:
                self.share_ptrs = []

                shares = ShareTensorCC.generate_shares(secret, self.session)
                for share, party in zip(shares, self.session.parties):
                    self.share_ptrs.append(share.send(party))

        elif shares is not None:
            self.share_ptrs = shares

    @staticmethod
    def sanity_checks(
        secret: Union[torch.Tensor, float, int],
        shape: Union[torch.Size, tuple],
        session: Session,
    ) -> Tuple[ShareTensor, Union[torch.Size, tuple], bool]:
        """Sanity check to validate that a new instance for ShareTensorCC can be
        created.

        :return: tuple representing the ShareTensor, the shape, if the secret
             is remote or local
        :rtype: tuple representing the ShareTensor (it
        """
        is_remote_secret = False

        if ispointer(secret):
            is_remote_secret = True
            if shape is None:
                raise ValueError(
                    "Shape must be specified if secret is at another party"
                )
        else:
            if isinstance(secret, (int, float)):
                secret = torch.tensor(data=[secret])

            if isinstance(secret, torch.Tensor):
                secret = ShareTensor(data=secret, session=session)

            shape = secret.shape

        return secret, shape, is_remote_secret

    @staticmethod
    def generate_przs(
        shape: Union[torch.Size, tuple], session: Session
    ) -> List[ShareTensor]:
        """Generate Pseudo-Random-Zero Shares at the parties involved in the computation

        :return: list of ShareTensor
        :rtype: list of PRZS
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
    def generate_shares(secret, session: Session) -> List[ShareTensor]:
        """Given a secret generate, split it into a number of shares such that
        each party would get one

        :return: list of shares
        :rtype: List of Zero Shares
        """
        if not isinstance(secret, ShareTensor):
            raise ValueError("Secret should be a ShareTensor")

        parties = session.parties
        nr_parties = len(parties)

        min_value = session.min_value
        max_value = session.max_value

        shape = secret.shape
        tensor_type = session.tensor_type

        random_shares = []
        generator = csprng.create_random_device_generator()

        for _ in range(nr_parties - 1):
            rand_value = torch.empty(size=shape, dtype=torch.long).random_(
                generator=generator
            )
            share = ShareTensor(session=session)

            # Add the share after such that we do not encode it
            share.tensor = rand_value
            random_shares.append(share)

        shares = []
        for i in range(len(parties)):
            if i == 0:
                share = random_shares[i]
            elif i < nr_parties - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]

            shares.append(share)
        return shares

    def reconstruct(self, decode: bool = True) -> torch.Tensor:
        """Request and get the shares from all the parties and reconstruct the secret.
        Depending on the value of "decode", the secret would be decoded or not using
        the FixedPrecision Encoder specific for the session

        :return: the secret reconstructed
        :rtype: tensor
        """

        def _request_and_get(share_ptr):
            """Function used to request and get a share - Duet Setup
            :return: the ShareTensor (local)
            :rtype: ShareTensor
            """
            share_ptr.request(block=True, verbose=True)
            res = share_ptr.get()
            return res

        def _get(share_ptr):
            """Function used to get a share - VM Setup
            :return: the ShareTensor (local)
            :rtype: ShareTensor
            """
            return share_ptr.get()

        request = None
        if isvm(self.session.parties[0]):
            request = _get
        else:
            # If not VirtualMachine, we are in Duet and we need to request
            # the data
            request = _request_and_get

        request = parallel_execution(request)

        args = [[share] for share in self.share_ptrs]
        local_shares = request(args)

        tensor_type = self.session.tensor_type
        plaintext = sum(local_shares)

        if decode:
            fp_encoder = FixedPointEncoder(
                base=self.session.config.encoder_base,
                precision=self.session.config.encoder_precision,
            )

            plaintext = fp_encoder.decode(plaintext.tensor)
        else:
            plaintext = plaintext.tensor

        return plaintext

    def add(
        self, y: Union["ShareTensorCC", torch.Tensor, float, int]
    ) -> "ShareTensorCC":
        """Apply the "add" operation between "self" and "y".

        :return: self + y
        :rtype: ShareTensorCC
        """
        return self.__apply_op(y, "add")

    def sub(
        self, y: Union["ShareTensorCC", torch.Tensor, float, int]
    ) -> "ShareTensorCC":
        """Apply the "sub" operation between "self" and "y".

        :return: self - y
        :rtype: ShareTensorCC
        """
        return self.__apply_op(y, "sub")

    def mul(
        self, y: Union["ShareTensorCC", torch.Tensor, float, int]
    ) -> "ShareTensorCC":
        """Apply the "mul" operation between "self" and "y".

        :return: self * y
        :rtype: ShareTensorCC
        """
        return self.__apply_op(y, "mul")

    def div(
        self, y: Union["ShareTensorCC", torch.Tensor, float, int]
    ) -> "ShareTensorCC":
        """Apply the "div" operation between "self" and "y".

        :return: self // y
        :rtype: ShareTensorCC
        """
        return self.__apply_op(y, "div")

    def __apply_private_op(self, y: "ShareTensorCC", op_str: str) -> "ShareTensorCC":
        """Apply an operation on 2 ShareTensorCC (secret shared values)

        :return: the operation "op_str" applied on "self" and "y"
        :rtype: ShareTensorCC
        """
        if y.session.uuid != self.session.uuid:
            raise ValueError(
                f"Need same session {self.session.uuid} and {y.session.uuid}"
            )

        if op_str in {"mul"}:
            from sympc.protocol.spdz import spdz

            shares = spdz.mul_master(self, y, op_str)
            result = ShareTensorCC(shares=shares, session=self.session)
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            result = ShareTensorCC(session=self.session)
            result.share_ptrs = [
                op(*share_tuple) for share_tuple in zip(self.share_ptrs, y.share_ptrs)
            ]

        return result

    def __apply_public_op(
        self, y: Union[torch.Tensor, float, int], op_str: str
    ) -> "ShareTensorCC":
        """Apply an operation on "self" which is a ShareTensorCC and a public value

        :return: the operation "op_str" applied on "self" and "y"
        :rtype: ShareTensorCC
        """

        op = getattr(operator, op_str)
        if op_str in {"mul"}:
            shares = [op(share, y) for share in self.share_ptrs]
        elif op_str in {"add", "sub"}:
            shares = self.share_ptrs
            # Only the rank 0 party has to add the element
            shares[0] = op(shares[0], y)
        else:
            raise ValueError(f"{op_str} not supported")

        result = ShareTensorCC(shares=shares, session=self.session)
        return result

    def __apply_op(
        self, y: Union["ShareTensorCC", torch.Tensor, float, int], op_str: str
    ) -> "ShareTensorCC":
        """Apply an operation on "self" which is a ShareTensorCC "y"
        This function checks if "y" is private or public value

        :return: the operation "op_str" applied on "self" and "y"
        :rtype: ShareTensorCC
        """
        is_private = isinstance(y, ShareTensorCC)

        if is_private:
            result = self.__apply_private_op(y, op_str)
        else:
            result = self.__apply_public_op(y, op_str)

        return result

    def __str__(self) -> str:
        """ Return the string representation of ShareTensorCC """
        type_name = type(self).__name__
        out = f"[{type_name}]"

        for share in self.share_ptrs:
            out = f"{out}\n\t| {share.client} -> {share.__name__}"
        return out

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
