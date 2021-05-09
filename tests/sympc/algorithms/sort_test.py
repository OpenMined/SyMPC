# third party
import pytest
import torch

from sympc.algorithms.algorithms import sort
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


@pytest.mark.parametrize("ascending", [True, False])
def test_mpc_sort(get_clients, ascending):
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=torch.tensor([1]), session=session)
    y = MPCTensor(secret=torch.tensor([3]), session=session)
    z = MPCTensor(secret=torch.tensor([6]), session=session)
    w = MPCTensor(secret=torch.tensor([18]), session=session)
    v = MPCTensor(secret=torch.tensor([5]), session=session)

    mpctensor_list = [x, y, z, w, v]

    sorted = sort(mpctensor_list, ascending=ascending)

    expected_list = [
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        torch.tensor([18.0]),
    ]

    sorted_list_1 = []
    for i in sorted:
        sorted_list_1.append(i.reconstruct())

    if ascending:
        assert sorted_list_1 == expected_list
    else:
        assert sorted_list_1 == expected_list[::-1]


def test_sort_invalidim_exception(get_clients):

    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=torch.tensor([1]), session=session)
    y = MPCTensor(secret=torch.tensor([3]), session=session)
    z = MPCTensor(secret=torch.tensor([6, 2]), session=session)

    mpctensor_list = [x, y, z]

    with pytest.raises(ValueError):
        sort(mpctensor_list)
