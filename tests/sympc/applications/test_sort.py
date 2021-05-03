# third party
import torch

from sympc.applications.sort import sort_mpctensor_list
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


def test_mpc_sort(get_clients):
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x = MPCTensor(secret=1, session=session)
    y = MPCTensor(secret=3, session=session)
    z = MPCTensor(secret=6, session=session)
    w = MPCTensor(secret=18, session=session)
    v = MPCTensor(secret=5, session=session)

    list = [x, y, z, w, v]

    ascending_sorted = sort_mpctensor_list(list)
    descending_sorted = sort_mpctensor_list(list, ascending=False)

    expected_list = [
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        torch.tensor([18.0]),
    ]

    sorted_list_1 = []
    for i in ascending_sorted:
        sorted_list_1.append(i.reconstruct())

    sorted_list_2 = []
    for i in descending_sorted:
        sorted_list_2.append(i.reconstruct())

    print(expected_list[::-1])
    print(sorted_list_2)

    assert sorted_list_1 == expected_list
    assert sorted_list_2 == expected_list[::-1]
