# third party
import torch

from sympc.protocol import FSS
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import ispointer


def test_share_tensor(get_clients) -> None:
    assert FSS.share_class == ShareTensor

    clients = get_clients(3)

    secret = torch.tensor([-1, 0, 1])
    shares = MPCTensor.generate_shares(secret, nr_parties=3)

    distributed_shares = FSS.distribute_shares(shares, parties=clients)

    assert all(ispointer(share_ptr) for share_ptr in distributed_shares)
