import torch
import torchcsprng as csprng

from typing import Union
from typing import List
from typing import Tuple
from typing import Any

from sympc.session import Session
from sympc.tensor.share import ShareTensor
from sympc.encoder import FixedPointEncoder
from sympc.utils import ispointer
from sympc.utils import isvm
from sympc.utils import parallel_execution

from copy import deepcopy
import operator


class ShareTensorCC:
    """
    This class is used by a party that wants to do some SMPC
    """

    __slots__ = {"share_ptrs", "session", "shape"}

    def __init__(
        self,
        session: Session,
        secret: Union[None, torch.Tensor, float, int] = None,
        shape: Union[None, torch.Size, tuple] = None,
        shares: Union[None, List[ShareTensor]] = None,
    ) -> None:

        if len(session.session_ptr) == 0:
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
                        self.share_ptrs[i] = self.share_ptr[i] + secret
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

        shape = tuple(shape)

        shares = []
        for session_ptr, generators_ptr in zip(
            session.session_ptr, session.przs_generators
        ):
            share_ptr = session_ptr.przs_generate_random_share(shape, generators_ptr)
            shares.append(share_ptr)

        return shares

    @staticmethod
    def generate_shares(secret, session: Session) -> List[ShareTensor]:
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

    def reconstruct(self, decode=True):
        def _request_and_get(share_ptr):
            share_ptr.request(block=True)
            return share_ptr.get()

        def _get(share_ptr):
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

    def add(self, y):
        return self.__apply_op(y, "add")

    def sub(self, y):
        return self.__apply_op(y, "sub")

    def mul(self, y):
        return self.__apply_op(y, "mul")

    def div(self, y):
        return self.__apply_op(y, "div")

    def __apply_private_op(self, y, op_str):
        if y.session.uuid != self.session.uuid:
            raise ValueError(
                f"Need same session {self.session.uuid} and {y.session.uuid}"
            )

        if op_str in {"mul"}:
            from ..protocol import spdz

            shares = spdz.mul_master(self, y, op_str)
            result = ShareTensorCC(shares=shares, session=self.session)
        elif op_str in {"sub", "add"}:
            op = getattr(operator, op_str)
            result = ShareTensorCC(session=self.session)
            result.share_ptrs = [
                op(*share_tuple) for share_tuple in zip(self.share_ptrs, y.share_ptrs)
            ]

        return result

    def __apply_public_op(self, y, op_str):
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

    def __apply_op(self, y, op):
        is_private = isinstance(y, ShareTensorCC)

        if is_private:
            result = self.__apply_private_op(y, op)
        else:
            result = self.__apply_public_op(y, op)

        return result

    def __str__(self):
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
